from overlay_measure.dialog_state_store import DialogStateStore


def test_dialog_state_remembers_files_and_folders(tmp_path):
    folder = tmp_path / "input"
    folder.mkdir()
    image = folder / "upper.png"
    image.write_bytes(b"image")
    store = DialogStateStore(tmp_path / "dialog_state.json")

    store.remember_file("image_import", str(image))
    assert store.directory("image_import") == str(folder)
    assert store.last_path("image_import") == str(image)

    second = tmp_path / "lower"
    second.mkdir()
    store.remember_directory("batch_folders", str(folder), [str(folder), str(second)])
    assert store.directory("batch_folders") == str(folder)
    assert store.paths("batch_folders") == [str(folder), str(second)]
