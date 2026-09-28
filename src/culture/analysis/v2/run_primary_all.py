#!/usr/bin/env python3
"""Run all primary-model steps in ONE process so the primary model is loaded once."""
import sys
import traceback


def main():
    steps = sys.argv[1:] or ["typology", "en_zh", "multi", "judge"]
    for s in steps:
        print(f"===== step {s}", flush=True)
        try:
            if s == "typology":
                import entity_typology as m; sys.argv = ["x"]; m.main()
            elif s == "en_zh":
                import entity_divergence_en_zh as m; sys.argv = ["x"]; m.main()
            elif s == "multi":
                import entity_divergence_multi as m; sys.argv = ["x"]; m.main()
            elif s == "judge":
                import meaning_clusters as m; sys.argv = ["x", "--judges", "primary"]; m.main()
        except Exception:
            traceback.print_exc()
        print(f"===== done {s}", flush=True)


if __name__ == "__main__":
    main()
