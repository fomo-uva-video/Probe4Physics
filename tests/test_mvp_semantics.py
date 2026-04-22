from __future__ import annotations

import unittest

from benchmarks.mvp.selection import (
    apply_mvp_selection,
    derive_binary_semantics,
    resolve_binary_choice_indices,
)


class MVPBinarySemanticsTests(unittest.TestCase):
    def test_derives_semantics_for_both_choice_orders(self) -> None:
        yes_first = derive_binary_semantics(("Yes", "No"), answer="Yes")
        self.assertEqual(yes_first.plausibility_label, 1)
        self.assertEqual(yes_first.yes_choice_idx, 0)
        self.assertEqual(yes_first.no_choice_idx, 1)

        no_first_plausible = derive_binary_semantics(("No", "Yes"), answer="Yes")
        self.assertEqual(no_first_plausible.plausibility_label, 1)
        self.assertEqual(no_first_plausible.yes_choice_idx, 1)
        self.assertEqual(no_first_plausible.no_choice_idx, 0)

        no_first_implausible = derive_binary_semantics(("No", "Yes"), answer="No")
        self.assertEqual(no_first_implausible.plausibility_label, 0)
        self.assertEqual(no_first_implausible.yes_choice_idx, 1)
        self.assertEqual(no_first_implausible.no_choice_idx, 0)

    def test_supports_long_form_aliases(self) -> None:
        possible = derive_binary_semantics(("Impossible", "Possible"), answer="Possible")
        self.assertEqual(possible.plausibility_label, 1)
        self.assertEqual(possible.yes_choice_idx, 1)
        self.assertEqual(possible.no_choice_idx, 0)

        plausible = derive_binary_semantics(("Implausible", "Plausible"), answer="Plausible")
        self.assertEqual(plausible.plausibility_label, 1)
        self.assertEqual(plausible.yes_choice_idx, 1)
        self.assertEqual(plausible.no_choice_idx, 0)

    def test_malformed_choices_fail_clearly(self) -> None:
        with self.assertRaisesRegex(ValueError, "Expected exactly 2 choices"):
            resolve_binary_choice_indices(("Yes",))

        with self.assertRaisesRegex(ValueError, "Expected exactly 2 choices"):
            derive_binary_semantics(None, answer="Yes")

    def test_non_binary_choices_fail_clearly(self) -> None:
        with self.assertRaisesRegex(ValueError, "Expected one yes-like and one no-like choice"):
            resolve_binary_choice_indices(("Left", "Right"))

        with self.assertRaisesRegex(ValueError, "Expected one yes-like and one no-like choice"):
            derive_binary_semantics(("Visible", "Hidden"), answer="Visible")

    def test_selection_accepts_mixed_choice_order_for_binary_plausibility_pairs(self) -> None:
        rows = [
            {
                "video_id": "pair_yes_first_0",
                "subset": "intuitive_physics",
                "source": "intphys",
                "video_path": "videos/pair_yes_first_0.mp4",
                "question": "Is this video physically plausible after the collision?",
                "candidates": ["Yes", "No"],
                "answer": "Yes",
            },
            {
                "video_id": "pair_yes_first_1",
                "subset": "intuitive_physics",
                "source": "intphys",
                "video_path": "videos/pair_yes_first_1.mp4",
                "question": "Is this video physically plausible after the collision?",
                "candidates": ["Yes", "No"],
                "answer": "No",
            },
            {
                "video_id": "pair_no_first_0",
                "subset": "intuitive_physics",
                "source": "intphys",
                "video_path": "videos/pair_no_first_0.mp4",
                "question": "Is this video physically plausible after the collision?",
                "candidates": ["No", "Yes"],
                "answer": "Yes",
            },
            {
                "video_id": "pair_no_first_1",
                "subset": "intuitive_physics",
                "source": "intphys",
                "video_path": "videos/pair_no_first_1.mp4",
                "question": "Is this video physically plausible after the collision?",
                "candidates": ["No", "Yes"],
                "answer": "No",
            },
        ]

        kept, dropped, report = apply_mvp_selection(
            rows,
            {
                "enabled": True,
                "subset": "intuitive_physics",
                "plausibility_only": True,
                "plausibility_keywords": ["is this video physically plausible"],
                "require_binary_yes_no": True,
                "yes_aliases": ["yes", "plausible", "possible"],
                "no_aliases": ["no", "implausible", "impossible"],
                "include_pair_ids": [],
                "exclude_pair_ids": [],
                "include_question_contains": [],
                "drop_incomplete_pairs": True,
                "artifacts": {"enabled": False},
            },
        )

        self.assertEqual(len(kept), 4)
        self.assertEqual(len(dropped), 0)
        self.assertEqual(report["kept_rows"], 4)


if __name__ == "__main__":
    unittest.main()
