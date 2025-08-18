def is_import_line(s: str) -> bool:
    return s.startswith("import ") or s.startswith("from ")


def _merge_maps(main_map, merge_map):
    for p, lst in merge_map.items():
        main_map.setdefault(p, []).extend(lst)
