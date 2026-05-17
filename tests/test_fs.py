from types import SimpleNamespace

from translatorhoi4.utils.fs import compute_output_path, rename_filename_for_lang


def test_rename_filename_collapses_compound_language_suffixes():
    assert rename_filename_for_lang("test_l_english.yml", "russian") == "test_l_russian.yml"
    assert rename_filename_for_lang("English_Russian.yml", "russian") == "russian.yml"
    assert rename_filename_for_lang("Kaiserreich_English.yml", "russian") == "Kaiserreich_russian.yml"
    assert rename_filename_for_lang("Kaiserreich_English_Russian.yml", "russian") == "Kaiserreich_russian.yml"
    assert rename_filename_for_lang("English_Kaiserreich.yml", "russian") == "russian_Kaiserreich.yml"


def test_compute_output_path_replaces_compound_language_folder(tmp_path):
    src_dir = tmp_path / "src"
    out_dir = tmp_path / "out"
    source_file = src_dir / "localisation" / "English_Russian" / "focus_l_english.yml"
    source_file.parent.mkdir(parents=True)
    source_file.write_text("l_english:\n", encoding="utf-8")

    cfg = SimpleNamespace(
        src_dir=str(src_dir),
        out_dir=str(out_dir),
        dst_lang="russian",
        in_place=False,
        use_mod_name=False,
        mod_name=None,
        rename_files=True,
    )

    assert compute_output_path(str(source_file), cfg) == str(
        out_dir / "localisation" / "russian" / "focus_l_russian.yml"
    )
