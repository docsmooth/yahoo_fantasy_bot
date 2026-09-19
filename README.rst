==================
Yahoo! Fantasy Bot
==================

A read-only assistant for a Yahoo! fantasy league.

Are you in a Yahoo! fantasy league with inactive managers?  Do you face teams
that start players on the IR, and have been for weeks?  Or do you need a few
more teams in your league?  This program helps alleviate that pain by
intelligently analyzing a Yahoo! fantasy team.  It can optimize a recommended lineup,
taking into consideration available players in the free agent pool.  Adjust the
IR and bench spots to account for star players that are a little banged up.
Evaluate proposed trades and recommend roster changes. It reads Yahoo data but
never changes Yahoo state; enter any recommended transactions manually. You
just run the program whenever you need to plan the lineup, which takes only a
few minutes to run.

Which tool do I want?
---------------------

The repo ships several entry points.  They do different jobs:

======================  ==============================================  ================
Tool                    What it does                                    When you use it
======================  ==============================================  ================
``ybot``                Recommends lineup, IL and trade decisions       In-season, daily
``ybot_setup``          One-time OAuth + config wizard                  First run only
``rank_players.py``     Ranks a draft pool from QuantHockey data        Before a draft
``draft_watcher.py``    Tails a live draft, notifies on each pick       During a draft
``mock_draft_test.py``  Simulates a snake draft off the rankings        Draft prep
======================  ==============================================  ================

``ybot`` and ``ybot_setup`` are installed onto your ``PATH`` by ``pip install``.
The three scripts under ``scripts/`` are not installed yet -- run them with
``python scripts/<name>.py`` from the repo root.  Making them proper console
scripts is tracked as an open issue.

Restrictions
------------

Yahoo Fantasy write operations are deliberately disabled. This accommodates
new Yahoo OAuth applications, which receive read-only Fantasy access. ``ybot``
still performs roster analysis and prints dry-run recommendations, but it
cannot apply roster changes or accept/reject trades.

This program will only optimize lineups for teams in a Yahoo! Head-to-Head
league.  It only works for teams in mlb or nhl leagues.

Note that the *scoring and ranking* tools (``rank_players.py``,
``mock_draft_test.py``) are NHL-only -- they read QuantHockey hockey exports.
The roster-management bot itself supports both mlb and nhl.

Installation
------------

You first need to set up the environment by installing the app.  You can pull
the latest from github

::

  git clone https://github.com/spilchen/yahoo_fantasy_bot.git
  cd yahoo_fantasy_bot
  python3 -m venv env
  source env/bin/activate
  pip install -r requirements.txt
  pip install -e .

Or you can simply install the package from pip.

::

  pip install yahoo-fantasy-bot

Python 3.8 or newer is required.  Optional fuzzy name matching (used when
merging Yahoo! points into the rankings) needs an extra:

::

  pip install -e '.[fuzzy]'

Configuration
-------------

Once the app is installed you need to set up the config file.  The config file
is what you pass to the bot.  It includes details about what Yahoo! league you
are going to run the bot against, the location of the file that holds the OAuth
credentials, and what league type it is.  There is a setup wizard that you can
run that will get you a working config file for your league.

Before you can run the setup wizard you will first need to request an API key
from Yahoo! from: https://developer.yahoo.com/apps/create.  The process is
quick.  New Yahoo apps receive read-only Fantasy access, which is sufficient for this
project. Upon completion you will be given a
consumer key and a consumer secret that you use with the setup wizard.

With key and secret, run the wizard like this:

::

  ybot_setup -k <consumer key> -s <consumer secret> oauth2.json my.cfg

``oauth2.json`` is used to store the credentials to access the team.  Using the
key and secret, it will pop up a webpage that will confirm you want to grant
access to the application.  It will give you a code, which you then paste back
into the window running the setup wizard.  The bearer token that it generates is
then saved in ``oauth2.json`` for all subsequent access.

Do not create ``oauth2.json`` by copying the example file: its token fields are
deliberate placeholders.  Start with ``ybot_setup -k <consumer key> -s
<consumer secret> --callback-uri <registered redirect URI> oauth2.json
my.cfg``; it writes the consumer credentials, completes the authorization
flow, and saves the access and refresh tokens. The callback URI must exactly
match a URI registered in the Yahoo developer console; otherwise Yahoo may
issue a token that cannot access Fantasy Sports resources.

Changing ``--callback-uri`` does not modify an existing credential file or
its tokens. Move the old ignored file aside and run setup again to authorize a
new token for the new callback URI.
Commands that access Yahoo validate all required fields first and name any
missing field without printing credential values.

Follow the rest of the prompts in the setup wizard.  Upon completion it will
write out a config file -- ``my.cfg`` in the example above.

.. warning::

   ``oauth2.json`` holds live credentials for your Yahoo! account.  It is
   covered by ``.gitignore`` (``oauth2*.json``) and must never be committed.
   ``oauth2.json.example`` shows the file's shape with placeholder values.

Execution
---------

Once installed and the config file created, you can run the program via this
command:

::

  ybot <cfg_file>

The script will choose a lineup based on available spots in the lineup and
print its proposed moves. It is always a dry run: ``--apply`` is rejected
before configuration, OAuth, or Yahoo API work begins. The prompt option has
no effect on writes because no writes are permitted. To get the full help text
use ``--help``.

Example
-------

Here is a sample run through.  In this run it will optimize the lineup, print
out the lineup then list the roster changes.  It will manage two players on the
IR and replace one player in the lineup from the free agent pool.

::

  $> ybot hockey.cfg
  Evaluating trades
  Adjusting lineup for player status
  Optimizing open lineup spots using available free agents
  100%|##############################################################|
  Optimizing lineup using players available from bench
  100%|##############################################################|
  Optimized lineup
  B   :                        WK_G G/A/PPP/SOG/PIM
  C   : Aleksander Barkov         3 38.0/63.0/32.0/241.0/10.0
  C   : Brayden Point             3 38.0/55.0/38.0/223.0/26.0
  LW  : Andrei Svechnikov         3 30.0/25.0/12.0/261.0/72.0
  LW  : Evander Kane              4 31.0/26.0/12.0/279.0/132.0
  RW  : David Pastrnak            3 44.0/53.0/39.0/281.0/40.0
  RW  : Alexander Radulov         3 28.0/45.0/24.0/212.0/64.0
  D   : Tyson Barrie              3 13.0/44.0/24.0/191.0/30.0
  D   : Thomas Chabot             3 15.0/43.0/15.0/197.0/36.0
  D   : P.K. Subban               4 12.0/40.0/16.0/174.0/70.0
  D   : Aaron Ekblad              3 14.0/25.0/11.0/186.0/55.0

  G   :                        WK_G W/SV%
  G   : Ben Bishop                3 31.0/0.922
  G   : Connor Hellebuyck         3 36.0/0.916

  Bench
  Jeff Skinner
  Patrice Bergeron

  Injury Reserve
  Sidney Crosby
  Mitchell Marner

  Computing roster moves to apply
  Move Sidney Crosby to IR
  Move Mitchell Marner to IR
  Add Brayden Point and drop Anthony Mantha
  Move David Pastrnak to RW
  Move Aleksander Barkov to C

Ranking players for a draft
---------------------------

If you have QuantHockey Excel exports in ``data/`` you can generate a ranked
players CSV.  The script reads one or more ``.xlsx`` files, computes a per-player
score normalized by games played (using empirical-Bayes shrinkage so that
small-sample players are pulled toward the league mean), and writes
``ranked_players.csv``.

::

  python scripts/rank_players.py --out ranked_players.csv

With no ``--input``, every ``data/*.xlsx`` file is discovered and combined.  The
newest season gets weight 1, the one before it ``decay``, and so on.  The
resolved order and weights are always printed so you can see what was used:

::

  Resolved file order (newest-first, --sort-by season) and decay weights (decay=0.5):
    [0] data/QuantHockey_2024-2025.xlsx  weight=1
    [1] data/QuantHockey_2023-2024.xlsx  weight=0.5

Commonly used options:

- ``--decay`` : per-file decay factor when combining multiple seasons (default 0.5)
- ``--k`` : prior weight for empirical-Bayes shrinkage (default 20).  Higher
  values pull small-sample players harder toward the league mean.
- ``--projected-games`` : games to project over for ranking (default 82)
- ``--sort-by`` : ``season`` (default, parsed from the filename), ``name`` or ``mtime``
- ``--normalize-file-weights`` : normalize per-file weights per-player so no one
  season dominates
- ``--goalie-input`` : path to a separate QuantHockey goalie export -- see
  `Goalies`_ below
- ``--fetch-yahoo`` : fetch Yahoo! points for comparison (needs ``--league-id``
  and ``--oauth-file``)

Run ``python scripts/rank_players.py --help`` for the full list.

The output CSV contains ``Name``, ``Team``, ``Pos``, and both per-season and
combined scores::

  Name,Team,Pos,gp_f0,shrunk_per_game_f0,projected_total_f0,raw_score_f0,
  goalie_stats_fabricated_f0,...,combined_shrunk_per_game,
  combined_projected_total,combined_ranking_score

Sort by ``combined_ranking_score`` to get the draft board.

Goalies
~~~~~~~

The standard QuantHockey *skater* export contains no goalie statistics -- no
wins, saves, goals against or shutouts.  Goalies appear in it as rows with
``Pos == 'G'`` and essentially empty stat lines.

Pass ``--goalie-input`` with a separate QuantHockey goalie export to score them
properly::

  python scripts/rank_players.py \
      --goalie-input data/QuantHockey_G_2024-2025.xlsx \
      --out ranked_players.csv

If you do not, goalies fall back to an estimate derived from games played, the
tool prints a loud warning naming the file and the number of affected players,
and the affected rows are flagged in the CSV with
``goalie_stats_fabricated = True``.  **Those numbers are not projections** --
do not draft off them.

Running scoring from ``ybot``
-----------------------------

``ybot --score`` runs the ranking step before managing the roster and makes the
resulting ``scored_players.csv`` available to the bot via the
``Scoring.scored_csv`` config key.  The bot merges ``combined_projected_total``
and ``combined_shrunk_per_game`` into its player pool.

::

  ybot --score --goalie-method gp-fallback --sort-by name my.cfg

``ybot`` re-exposes the ``rank_players.py`` scoring flags and forwards them
through, so anything in the list above can be given directly to ``ybot``.  A
contract test keeps the two flag lists in sync.  You can also pass arbitrary
extra arguments through with ``--rank-extra="..."``.

If the scoring step fails, ``ybot`` aborts rather than silently managing your
roster with no scored data.  Pass ``--continue-on-score-failure`` to override.

Watching a live draft
---------------------

See `README_DRAFT_WATCHER.md <README_DRAFT_WATCHER.md>`_ for the draft watcher,
which polls Yahoo! during a live draft and notifies you as picks come in.

Development
-----------

Run the test suite from the repo root::

  pytest

Tests live in ``tests/``.  ``tests/fixtures/quanthockey_sample.xlsx`` is a small
fixture carved from a real QuantHockey export -- it reproduces the real
50-column schema on purpose.  Write new scoring tests against it rather than
hand-inventing column names; several long-lived scoring bugs survived precisely
because the older tests used invented column names that did not match the real
data.

Regenerate the fixtures with::

  python tests/fixtures/make_fixture.py
  python tests/fixtures/make_goalie_fixture.py

Known limitations
-----------------

- **Goalie rankings need a separate data file.**  See `Goalies`_ above.  Without
  ``--goalie-input``, goalie scores are fabricated from games played and are
  flagged as such.
- **Scoring is NHL-only.**  ``scoring.py`` reads QuantHockey hockey exports.  The
  roster bot supports mlb and nhl, but there is no mlb ranking path.
- **Hits are not scored.**  The QuantHockey export has a ``HITS`` column; it is
  deliberately not part of the default weights.
- **There is no sample config file.**  ``ybot``'s help text refers to
  ``sample_config.ini``, which does not exist yet.  Use ``ybot_setup`` to
  generate a working config.
- **Goalies are exempt from shrinkage.**  Once real goalie stats are supplied, a
  goalie with very few games can still rank too highly.
