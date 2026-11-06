import json

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pytest

from scripts import export_numeric_attention_reports as numeric


def test_additive_node_effect_and_pair_specific_residual_are_distinguished():
    effect = np.array([5., 0., 0., 0., 0., 0.])
    r = 2 + effect[:, None] + effect[None, :]
    np.fill_diagonal(r, 0)
    description, component, residual = numeric.node_description(r)
    assert description["node_component_variation_fraction"] == pytest.approx(1)
    assert description["broadly_high_nodes"] == [0]
    np.testing.assert_allclose(residual, 0, atol=1e-12)
    for row in description["nodes"]:
        assert row["row_mean"] == row["column_mean"]
    # A regular cycle has pair-specific variation but equal row/column means.
    cycle = np.zeros((6, 6))
    for i in range(6):
        cycle[i, (i + 1) % 6] = cycle[(i + 1) % 6, i] = 1
    cycle_description, _, _ = numeric.node_description(cycle)
    assert cycle_description["broadly_high_nodes"] == []
    assert cycle_description["node_component_variation_fraction"] == pytest.approx(0, abs=1e-12)
    constant = np.full((6, 6), .1)
    np.fill_diagonal(constant, 0)
    constant_description, _, _ = numeric.node_description(constant)
    assert constant_description["broadly_high_nodes"] == []
    assert np.isnan(constant_description["node_component_variation_fraction"])


@pytest.mark.parametrize("values,category,third,center", [
    ([.1, .2, .3], "no_pairs_high", None, None),
    ([.9, .2, .3], "one_pair_high", 2, None),
    ([.9, .8, .3], "two_pairs_high", None, 0),
    ([.9, .8, .7], "all_three_pairs_high", None, None),
])
def test_triad_categories_preserve_all_three_values(values, category, third, center):
    r = np.zeros((3, 3))
    for (i, j), value in zip(((0, 1), (0, 2), (1, 2)), values):
        r[i, j] = r[j, i] = value
    row = numeric.triad_values(r, np.array([[0, 1, 2]]), .5)[0]
    assert row["category"] == category
    assert row["third_node"] == third and row["shared_center_node"] == center
    assert row["min"] == min(values) and row["max"] == max(values)
    assert row["mean"] == pytest.approx(np.mean(values))
    assert [row[k] for k in ("R_ij", "R_ik", "R_jk")] == values


def test_random_nonedges_are_reproducible_and_focus_uses_true_edges():
    a2 = np.zeros((30, 30), dtype=int)
    a2[:10, :10] = 1
    np.fill_diagonal(a2, 0)
    first, seed = numeric.nonedge_sample(a2, "sample0", 42, 20)
    repeated, _ = numeric.nonedge_sample(a2, "sample0", 42, 20)
    np.testing.assert_array_equal(first, repeated)
    assert len(first) == len(np.unique(first, axis=0)) == 20
    assert not a2[first[:, 0], first[:, 1]].any()
    assert numeric.focus_nodes(a2) == list(range(10))


def test_numeric_image_contains_text_without_colored_cells(tmp_path):
    fig, ax = plt.subplots()
    numeric.draw_numeric(ax, np.array([[0, .1234], [.1234, 0]]), [4, 9], "test", decimals=3)
    assert [text.get_text() for text in ax.texts] == ["0.000", "0.123", "0.123", "0.000"]
    assert len(ax.images) == 0
    path = tmp_path / "numeric.png"
    fig.savefig(path)
    plt.close(fig)
    pixels = mpimg.imread(path)
    np.testing.assert_array_equal(pixels[..., 0], pixels[..., 1])
    np.testing.assert_array_equal(pixels[..., 1], pixels[..., 2])


@pytest.fixture
def numeric_source(tmp_path):
    root = tmp_path / "input"
    folder = root / "example"
    folder.mkdir(parents=True)
    a2 = np.zeros((30, 30), dtype=np.uint8)
    a2[0, 1] = a2[1, 0] = a2[1, 2] = a2[2, 1] = 1
    np.save(folder / "A2_true.npy", a2)
    h3 = np.zeros((30, 1), dtype=np.uint8)
    h3[:3, 0] = 1
    np.save(folder / "H3_true.npy", h3)
    (folder / "triads_true.json").write_text("[[0,1,2]]")
    rng = np.random.default_rng(0)
    r = rng.normal(size=(2, 4, 30, 30)).astype(np.float32)
    r = (r + r.swapaxes(-1, -2)) / 2
    r[..., np.arange(30), np.arange(30)] = 0
    a = np.abs(r) * .01
    np.savez_compressed(folder / "attention_matrices.npz", symmetric_logits=r,
                        symmetric_attention=a, trajectory_id="example")
    return root, folder, r, a


def test_reports_export_all_heads_and_all_true_edges(numeric_source, tmp_path, monkeypatch):
    root, source, logits, attention = numeric_source
    plots = []
    monkeypatch.setattr(numeric, "numeric_image", lambda matrix, path, *args, **kwargs: plots.append(path.name))
    monkeypatch.setattr(numeric, "focus_image", lambda entry, nodes, path, *args: plots.append(path.name))
    output = numeric.export_numeric(root, tmp_path / "output", sample="example")
    folder = output / "example"
    assert len(list(folder.glob("*.csv"))) == 17
    assert len(plots) == 19
    np.testing.assert_array_equal(np.loadtxt(folder / "A2_true.csv", delimiter=","), np.load(source / "A2_true.npy"))
    np.testing.assert_allclose(np.loadtxt(folder / "layer1_head1_symmetric_logits.csv", delimiter=","),
                               logits[0, 0], atol=1e-9, rtol=1e-8)
    summary = json.loads((folder / "numeric_summary.json").read_text())
    assert len(summary["heads"]) == 16 and summary["n_true_pairs"] == 2
    assert summary["n_true_triads"] == 1 and len(summary["sampled_nonedges"]) == 20
    report = (folder / "pair_value_report.txt").read_text()
    assert report.count("所有真实二元边") == 16
    assert report.count("固定随机抽取的非边") == 16
    assert all(len(head["nodes"]) == 30 for head in summary["heads"])
    assert all(len(head["triad_values"]) == 1 for head in summary["heads"])
    with pytest.raises(FileExistsError):
        numeric.export_numeric(root, output)


def test_empty_triads_and_malformed_incidence(numeric_source):
    _, source, _, _ = numeric_source
    np.save(source / "H3_true.npy", np.zeros((30, 0), dtype=np.uint8))
    (source / "triads_true.json").write_text("[]")
    entry = numeric.read_sample(source)
    assert entry["triads"].shape == (0, 3)
    assert numeric.triad_values(entry["symmetric_logits"][0, 0], entry["triads"], .5) == []
    np.save(source / "H3_true.npy", np.ones((30, 1), dtype=np.uint8))
    with pytest.raises(ValueError, match="H3 columns"):
        numeric.read_sample(source)
