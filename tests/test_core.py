"""Unit tests for the parts that must be right for the results to mean anything.

    python -m pytest tests/ -q          # or: python tests/test_core.py

These are torch-free by design, so they run in CI without a GPU. The
model-dependent paths (SFT, GRPO, latent, steering) are covered by
``scripts/smoke_test.py --tier 2``.
"""
from __future__ import annotations

import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from latentsql.config import load_config
from latentsql.data.synthetic import load_synthetic
from latentsql.db.execute import execute_sql, execution_accuracy, gold_is_ordered, is_read_only
from latentsql.db.schema import load_schema, render_schema
from latentsql.db.value_index import ValueIndex
from latentsql.distill.rewards import RewardSchedule, group_advantages
from latentsql.mas.agents import mask_question, query_skeleton, _group_by_execution
from latentsql.mas.trace import (
    AgentTurn, DebateTrace, extract_final_sql, format_score, normalize_sql,
    parse_trace, tag_order_score,
)

_WORK = tempfile.mkdtemp(prefix="latentsql_tests_")
_EXAMPLES = load_synthetic(_WORK)
_DB = _EXAMPLES[0].db_path


class TestExecution(unittest.TestCase):
    def test_gold_queries_execute(self):
        for e in _EXAMPLES:
            r = execute_sql(e.db_path, e.gold_sql, timeout=10)
            self.assertTrue(r.ok, f"{e.qid}: {r.error}")

    def test_read_only_guard(self):
        self.assertTrue(is_read_only("SELECT * FROM t"))
        self.assertTrue(is_read_only("WITH x AS (SELECT 1) SELECT * FROM x"))
        self.assertFalse(is_read_only("DROP TABLE t"))
        self.assertFalse(is_read_only("SELECT 1; DELETE FROM t"))
        self.assertFalse(is_read_only("ATTACH DATABASE 'x' AS y"))
        # a comment must not smuggle a write past the guard
        self.assertFalse(is_read_only("SELECT 1 /* ok */ ; UPDATE t SET a=1"))

    def test_order_sensitivity(self):
        self.assertTrue(gold_is_ordered("SELECT a FROM t ORDER BY a"))
        self.assertFalse(gold_is_ordered("SELECT a FROM t"))
        # ORDER BY inside a sub-query does not order the outer result
        self.assertFalse(
            gold_is_ordered("SELECT a FROM t WHERE b IN (SELECT c FROM u ORDER BY c)")
        )

    def test_execution_accuracy_semantics(self):
        # same rows, different column order -> still a match (Spider semantics)
        ok, _, _ = execution_accuracy(
            _DB, "SELECT name, capacity FROM stadium", "SELECT capacity, name FROM stadium"
        )
        self.assertTrue(ok)
        # different rows -> no match
        ok, _, _ = execution_accuracy(
            _DB, "SELECT name FROM stadium LIMIT 1", "SELECT name FROM stadium"
        )
        self.assertFalse(ok)
        # numeric vs string representation of the same value -> match
        ok, _, _ = execution_accuracy(_DB, "SELECT '5'", "SELECT 5")
        self.assertTrue(ok)

    def test_unordered_comparison_ignores_row_order(self):
        ok, _, _ = execution_accuracy(
            _DB,
            "SELECT name FROM stadium ORDER BY name DESC",
            "SELECT name FROM stadium",
        )
        self.assertTrue(ok)

    def test_ordered_comparison_respects_row_order(self):
        ok, _, _ = execution_accuracy(
            _DB,
            "SELECT name FROM stadium ORDER BY capacity ASC",
            "SELECT name FROM stadium ORDER BY capacity DESC",
        )
        self.assertFalse(ok)

    def test_bad_sql_is_an_error_not_a_crash(self):
        r = execute_sql(_DB, "SELECT nope FROM nowhere")
        self.assertFalse(r.ok)
        self.assertIn("Error", r.error)


class TestSchema(unittest.TestCase):
    def test_introspection(self):
        s = load_schema(_DB, "concert_singer")
        self.assertEqual(len(s.tables), 4)
        self.assertTrue(s.foreign_keys)
        self.assertIsNotNone(s.table("singer").column("country"))

    def test_render_includes_keys_and_examples(self):
        text = render_schema(load_schema(_DB), style="code")
        self.assertIn("CREATE TABLE", text)
        self.assertIn("PRIMARY KEY", text)
        self.assertIn("FOREIGN KEY", text)
        self.assertIn("examples:", text)

    def test_prune_keeps_join_keys(self):
        s = load_schema(_DB)
        pruned = s.prune({"concert": ["concert_name"], "stadium": ["name"]})
        self.assertEqual(len(pruned.tables), 2)
        # the FK columns must survive or the pruned schema is unjoinable
        self.assertIsNotNone(pruned.table("concert").column("stadium_id"))
        self.assertIsNotNone(pruned.table("stadium").column("stadium_id"))

    def test_prune_with_empty_links_is_identity(self):
        s = load_schema(_DB)
        self.assertEqual(len(s.prune({}).tables), len(s.tables))


class TestValueIndex(unittest.TestCase):
    def test_exact_and_fuzzy_retrieval(self):
        idx = ValueIndex.build(_DB, "concert_singer")
        hits = idx.search("France", top_k=5, min_similarity=0.6)
        self.assertTrue(any(h.value == "France" for h in hits))
        # missing letter: trigrams still overlap
        self.assertTrue(idx.search("Frace", top_k=5, min_similarity=0.5))
        # transposition: shares no trigram with "France", so this exercises
        # the prefix fallback rather than the MinHash index
        typo = idx.search("Frnace", top_k=5, min_similarity=0.5)
        self.assertTrue(typo, "transposed keyword returned nothing")
        self.assertTrue(any(h.value == "France" for h in typo), [h.value for h in typo])

    def test_numeric_columns_are_not_indexed(self):
        idx = ValueIndex.build(_DB, "concert_singer")
        cols = {c for _t, c, _v in idx.lsh.entries}
        self.assertNotIn("capacity", cols)
        self.assertIn("name", cols)


class TestTrace(unittest.TestCase):
    def _trace(self) -> DebateTrace:
        return DebateTrace(
            qid="q1", db_id="d", question="How many stadiums?",
            gold_sql="SELECT count(*) FROM stadium",
            schema_links="Tables: stadium",
            turns=[
                AgentTurn("divide_conquer", 0, "divide then conquer",
                          "SELECT count(*) FROM stadium", exec_ok=True),
                AgentTurn("query_plan", 0, "scan stadium", "SELECT count(*) FROM stadium",
                          exec_ok=True),
            ],
            consensus_sql="SELECT count(*) FROM stadium",
            consensus_reached=True, n_rounds=1,
        )

    def test_render_parse_roundtrip(self):
        text = self._trace().render()
        p = parse_trace(text)
        self.assertEqual(p["n_rounds"], 1)
        self.assertEqual(sorted(p["agent_names"]), ["divide_conquer", "query_plan"])
        self.assertEqual(p["final_sql"], "SELECT count(*) FROM stadium")
        self.assertTrue(p["has_end"] and p["has_consensus"])

    def test_format_score(self):
        self.assertEqual(format_score(self._trace().render()), 1.0)
        self.assertEqual(format_score("no tags here"), 0.0)
        self.assertAlmostEqual(format_score("<|schema_link|> <|endofdebate|>"), 0.5)

    def test_tag_order_score_penalises_scrambling(self):
        good = self._trace().render()
        self.assertEqual(tag_order_score(good), 1.0)
        scrambled = "<|endofdebate|> <|consensus|> <|round 1|> <|schema_link|>"
        self.assertEqual(tag_order_score(scrambled), 0.0)

    def test_extract_final_sql_handles_internalised_output(self):
        # fully tagged
        self.assertEqual(
            extract_final_sql("<|consensus|>\n```sql\nSELECT 1\n```\n<|/consensus|>"), "SELECT 1"
        )
        # tags dropped after RL - the point of internalisation
        self.assertEqual(extract_final_sql("```sql\nSELECT 2\n```"), "SELECT 2")
        # bare query, no fence
        self.assertEqual(extract_final_sql("The answer is SELECT 3 FROM t"), "SELECT 3 FROM t")
        # nothing usable
        self.assertIsNone(extract_final_sql("I don't know."))

    def test_extract_prefers_consensus_over_earlier_agent_sql(self):
        text = (
            "<|round 1|><|agent query_plan|>```sql\nSELECT wrong\n```<|/agent|><|/round|>"
            "<|consensus|>```sql\nSELECT right\n```<|/consensus|><|endofdebate|>"
        )
        self.assertEqual(extract_final_sql(text), "SELECT right")

    def test_normalize_sql(self):
        self.assertEqual(normalize_sql("  SELECT   1 ;  "), "SELECT 1")
        self.assertEqual(normalize_sql("```sql\nSELECT 1\n```"), "SELECT 1")
        self.assertEqual(normalize_sql(None), "")


class TestRewards(unittest.TestCase):
    def setUp(self):
        self.cfg = load_config().grpo.reward

    def test_weight_and_length_anneal_monotonically(self):
        s = RewardSchedule(self.cfg, total_steps=50)
        w = [s.w_fmt(i) for i in range(50)]
        l = [s.length_limit(i) for i in range(50)]
        self.assertEqual(w[0], self.cfg.w_fmt_start)
        self.assertAlmostEqual(w[-1], self.cfg.w_fmt_end)
        self.assertTrue(all(a >= b for a, b in zip(w, w[1:])), "w_fmt must not increase")
        self.assertTrue(all(a >= b for a, b in zip(l, l[1:])), "length limit must not increase")
        self.assertEqual(l[0], self.cfg.len_start)
        self.assertEqual(l[-1], self.cfg.len_end)

    def test_schedule_is_stable_for_a_single_step(self):
        s = RewardSchedule(self.cfg, total_steps=1)
        self.assertAlmostEqual(s.w_fmt(0), self.cfg.w_fmt_end)

    def test_group_advantages(self):
        a = group_advantages([1.0, 0.0, 0.0, 0.0])
        self.assertAlmostEqual(sum(a), 0.0, places=6)
        self.assertGreater(a[0], 0)
        self.assertLess(a[1], 0)
        self.assertEqual(group_advantages([0.3] * 5), [0.0] * 5)
        self.assertEqual(group_advantages([]), [])


class TestDailSelection(unittest.TestCase):
    def test_masking_removes_domain_words(self):
        s = load_schema(_DB)
        masked = mask_question("How many singers are from France?", s)
        self.assertNotIn("singer", masked)
        self.assertIn("<mask>", masked)

    def test_query_skeleton(self):
        sk = query_skeleton("SELECT count(*) FROM a JOIN b ON a.id=b.id GROUP BY x")
        self.assertIn("join", sk)
        self.assertIn("group by", sk)
        self.assertNotIn("order by", sk)


class TestSelectorGrouping(unittest.TestCase):
    def test_identical_results_group_together(self):
        from latentsql.mas.agents import Context

        ctx = Context(question="q", db_path=_DB)
        turns = [
            AgentTurn("a", 0, "", "SELECT count(*) FROM stadium"),
            AgentTurn("b", 0, "", "SELECT COUNT(*) FROM stadium"),   # same result
            AgentTurn("c", 0, "", "SELECT name FROM stadium"),       # different
        ]
        groups = _group_by_execution(ctx, turns)
        sizes = sorted(len(g) for g in groups)
        self.assertEqual(sizes, [1, 2])


class TestMixedDataset(unittest.TestCase):
    """The combined BIRD + Spider set (``dataset: mix``)."""

    def test_every_prefix_holds_the_configured_ratio(self):
        from latentsql.data.loaders import Example, dataset_counts, interleave_by_ratio

        bird = [Example(qid=f"b{i}", db_id="d", question="q", gold_sql="s",
                        dataset="bird") for i in range(60)]
        spider = [Example(qid=f"s{i}", db_id="d", question="q", gold_sql="s",
                          dataset="spider") for i in range(20)]
        merged = interleave_by_ratio(bird, spider)

        self.assertEqual(dataset_counts(merged), {"bird": 60, "spider": 20})
        # slicing with --limit must not silently turn into a BIRD-only run
        for k in (8, 20, 40, 80):
            counts = dataset_counts(merged[:k])
            self.assertAlmostEqual(counts.get("bird", 0) / k, 0.75, delta=0.06)

    def test_interleave_is_deterministic(self):
        from latentsql.data.loaders import Example, interleave_by_ratio

        a = [Example(qid=f"a{i}", db_id="d", question="q", gold_sql="s") for i in range(7)]
        b = [Example(qid=f"b{i}", db_id="d", question="q", gold_sql="s") for i in range(3)]
        first = [e.qid for e in interleave_by_ratio(a, b)]
        self.assertEqual(first, [e.qid for e in interleave_by_ratio(a, b)])

    def test_git_lfs_pointers_are_not_mistaken_for_data(self):
        """An unfetched LFS pointer is a small text file that json.load chokes on."""
        from latentsql.data.loaders import _is_placeholder

        root = tempfile.mkdtemp(prefix="latentsql_lfs_")
        pointer = os.path.join(root, "dev.json")
        with open(pointer, "w", encoding="utf-8") as fh:
            fh.write("version https://git-lfs.github.com/spec/v1\n"
                     "oid sha256:9b4c38da7f07\nsize 780246\n")
        empty = os.path.join(root, "empty.json")
        open(empty, "w").close()
        real = os.path.join(root, "real.json")
        with open(real, "w", encoding="utf-8") as fh:
            fh.write("[]")

        self.assertTrue(_is_placeholder(pointer))
        self.assertTrue(_is_placeholder(empty))
        self.assertFalse(_is_placeholder(real))

    def test_db_root_follows_the_split(self):
        """BIRD ships dev/ and train/ databases side by side."""
        from latentsql.data.loaders import find_db_root

        root = tempfile.mkdtemp(prefix="latentsql_split_")
        for sub in ("dev/dev_databases/x", "train/train_databases/x"):
            os.makedirs(os.path.join(root, sub), exist_ok=True)
        self.assertTrue(find_db_root(root, "bird", "dev").endswith("dev_databases"))
        self.assertTrue(find_db_root(root, "bird", "train").endswith("train_databases"))

    def test_gold_sql_is_never_a_stringified_parse_tree(self):
        """Canonical Spider dev.json puts the parsed tree in `sql`, the text in `query`.

        Stringifying that dict yields a repr that cannot execute, so every
        Spider question scores as wrong and the failure is completely silent.
        """
        from latentsql.data.loaders import _get

        spider_dev = {
            "db_id": "concert_singer",
            "question": "How many singers do we have?",
            "query": "SELECT count(*) FROM singer",
            "sql": {"from": {"table_units": [["table_unit", 1]]}, "select": [False, []]},
        }
        self.assertEqual(
            _get(spider_dev, "SQL", "sql", "query", "gold_sql"),
            "SELECT count(*) FROM singer",
        )

        # the FusionSQL dumps keep the string under `sql`; that must still win
        fusion = {"db_id": "x", "question": "q", "sql": "SELECT 1"}
        self.assertEqual(_get(fusion, "SQL", "sql", "query", "gold_sql"), "SELECT 1")

        # non-string scalars are still coerced (BIRD question_id is an int)
        self.assertEqual(_get({"question_id": 42}, "question_id"), "42")

    def test_report_splits_ex_by_benchmark(self):
        from latentsql.eval.metrics import Prediction, build_report

        preds = [
            Prediction(qid="1", db_id="d", db_path="", question="q", gold_sql="g",
                       pred_sql="p", correct=True, dataset="bird"),
            Prediction(qid="2", db_id="d", db_path="", question="q", gold_sql="g",
                       pred_sql="p", correct=False, dataset="bird"),
            Prediction(qid="3", db_id="d", db_path="", question="q", gold_sql="g",
                       pred_sql="p", correct=True, dataset="spider"),
            Prediction(qid="4", db_id="d", db_path="", question="q", gold_sql="g",
                       pred_sql="p", correct=True, dataset="spider"),
        ]
        rep = build_report(preds, method="student", dataset="mix")
        self.assertAlmostEqual(rep.ex, 0.75)
        self.assertAlmostEqual(rep.by_dataset["bird"]["ex"], 0.5)
        self.assertAlmostEqual(rep.by_dataset["spider"]["ex"], 1.0)


class TestSystemPanel(unittest.TestCase):
    """The four published systems used as debate panellists."""

    def test_all_four_systems_are_registered_as_agents(self):
        from latentsql.mas.agents import AGENT_REGISTRY

        for name in ("din_sql", "dail_sql", "chess", "chase_sql"):
            self.assertIn(name, AGENT_REGISTRY)
            self.assertTrue(AGENT_REGISTRY[name].system_fn)

    def test_system_tags_are_in_the_vocabulary(self):
        from latentsql.mas.trace import agent_tag, special_tokens

        toks = special_tokens()
        for name in ("din_sql", "dail_sql", "chess", "chase_sql"):
            self.assertIn(agent_tag(name), toks)

    def test_every_baseline_has_a_turn_twin(self):
        from latentsql.mas import baselines

        for name in ("din_sql", "dail_sql", "chess", "chase_sql"):
            self.assertTrue(callable(getattr(baselines, f"{name}_turn")))

    def test_pick_recovers_the_reasoning_behind_a_selection(self):
        from latentsql.mas.agents import Context
        from latentsql.mas.baselines import _pick

        class _Sel:
            sql = "SELECT COUNT(*) FROM stadium"

        ctx = Context(question="q", db_path=_DB)
        turns = [
            AgentTurn("query_plan", 0, "counted the rows", "SELECT count(*) FROM stadium"),
            AgentTurn("dail_icl", 0, "listed names", "SELECT name FROM stadium"),
        ]
        picked = _pick(ctx, "chess", _Sel(), turns, round_index=0)
        self.assertEqual(picked.agent, "chess")
        self.assertEqual(picked.reasoning, "counted the rows")


class TestDenseRewards(unittest.TestCase):
    """Atomic-op partial credit (FINER-SQL), which keeps GRPO groups non-degenerate."""

    GOLD = "SELECT COUNT(DISTINCT name) FROM director WHERE director = 'Reitherman'"

    def setUp(self):
        from latentsql.distill.dense_rewards import AtomicOpsReward

        self.a = AtomicOpsReward()

    def test_aliasing_and_case_do_not_change_the_score(self):
        variant = ("select count(distinct T1.name) from director AS T1 "
                   "where T1.director = 'Reitherman'")
        self.assertAlmostEqual(self.a.jaccard(variant, self.GOLD), 1.0)

    def test_near_misses_are_graded_not_binary(self):
        """The whole point: a wrong query must not always score the same as gold."""
        exact = self.a.jaccard(self.GOLD, self.GOLD)
        no_distinct = self.a.jaccard(
            "SELECT COUNT(name) FROM director WHERE director = 'Reitherman'", self.GOLD)
        no_where = self.a.jaccard(
            "SELECT COUNT(DISTINCT name) FROM director", self.GOLD)
        unrelated = self.a.jaccard("SELECT COUNT(*) FROM movie", self.GOLD)

        self.assertEqual(exact, 1.0)
        self.assertLess(no_distinct, exact)       # DISTINCT lives inside COUNT()
        self.assertLess(no_where, exact)
        self.assertEqual(unrelated, 0.0)
        self.assertGreater(no_where, unrelated)   # closer is worth more

    def test_unparseable_sql_scores_zero_rather_than_raising(self):
        self.assertEqual(self.a.jaccard("SELECT COUNT( FROM", self.GOLD), 0.0)
        self.assertEqual(self.a.score(None, self.GOLD), 0.0)

    def test_shaping_is_monotone_and_bounded(self):
        prev = -1.0
        for sql in ("SELECT COUNT(*) FROM movie",
                    "SELECT COUNT(DISTINCT name) FROM director",
                    self.GOLD):
            v = self.a.score(sql, self.GOLD)
            self.assertGreaterEqual(v, 0.0)
            self.assertLessEqual(v, 1.0)
            self.assertGreater(v, prev)
            prev = v

    def test_dense_terms_only_reward_wrong_rollouts(self):
        """A correct rollout must not be able to earn extra by resembling things."""
        from latentsql.config import RewardConfig
        from latentsql.distill.rewards import DebateReward, ExecutionOracle, RewardSchedule

        cfg = RewardConfig()
        cfg.w_atomic = 1.0
        sched = RewardSchedule(cfg, total_steps=10)

        class _Oracle(ExecutionOracle):
            def __init__(self, verdict):
                super().__init__(timeout=1.0)
                self.verdict = verdict

            def is_correct(self, db_path, pred_sql, gold_sql):
                return self.verdict

            def executes(self, db_path, pred_sql):
                return True

        text = f"<|consensus|>\n```sql\n{self.GOLD}\n```\n<|/consensus|>"
        right = DebateReward(cfg, _Oracle(True), sched)(text, None, "", self.GOLD, 0)
        wrong = DebateReward(cfg, _Oracle(False), sched)(text, None, "", self.GOLD, 0)

        self.assertEqual(right.r_atomic, 0.0)     # already fully credited
        self.assertGreater(wrong.r_atomic, 0.0)   # graded partial credit


class TestBrevityReward(unittest.TestCase):
    """Breaks the saturated-group tie GRPO otherwise cannot learn from."""

    GOLD = "SELECT COUNT(*) FROM t"

    def _reward(self, correct, n_tokens, w_brevity=0.5):
        from latentsql.config import RewardConfig
        from latentsql.distill.rewards import DebateReward, ExecutionOracle, RewardSchedule

        class _O(ExecutionOracle):
            def __init__(self, v):
                super().__init__(timeout=1.0)
                self.v = v

            def is_correct(self, *a):
                return self.v

            def executes(self, *a):
                return True

        cfg = RewardConfig()
        cfg.w_brevity = w_brevity
        cfg.brevity_ref_tokens = 1024
        text = f"<|consensus|>\n```sql\n{self.GOLD}\n```\n<|/consensus|>"
        return DebateReward(cfg, _O(correct), RewardSchedule(cfg, total_steps=10),
                            decode=lambda x: text)(text, list(range(n_tokens)),
                                                   "", self.GOLD, 0)

    def test_shorter_correct_rollout_scores_higher(self):
        short = self._reward(True, 200)
        long = self._reward(True, 900)
        self.assertEqual(short.r_clip, long.r_clip)      # both fully correct
        self.assertGreater(short.total, long.total)      # but not equally rewarded

    def test_wrong_rollouts_earn_no_brevity_credit(self):
        """Otherwise the model learns to answer briefly and wrongly."""
        self.assertEqual(self._reward(False, 50).r_brevity, 0.0)
        self.assertGreater(self._reward(True, 50).r_brevity, 0.0)

    def test_disabled_by_default(self):
        self.assertEqual(self._reward(True, 200, w_brevity=0.0).r_brevity, 0.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
