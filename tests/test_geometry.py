from collections import Counter

import pytest
import torch

from watergym.geometry import Mesh, foil_mesh, load_mesh, pod, sail, signed_volume, tube
from watergym.vessels import bluerov2, box_barge, moth, otter

VESSELS = [moth, otter, bluerov2, box_barge]


def unpaired_edges(mesh: Mesh) -> int:
    """Directed edges without a reversed twin, after merging vertices at the same spot.
    Zero means every part is closed and wound consistently."""
    _, welded = torch.unique((mesh.vertices * 1e5).round(), dim=0, return_inverse=True)
    triangles = welded[mesh.triangles]
    a, b, c = triangles.unbind(1)
    triangles = triangles[(a != b) & (b != c) & (c != a)]
    edges = Counter(
        map(
            tuple,
            torch.cat((triangles[:, :2], triangles[:, 1:], triangles[:, ::2].flip(1))).tolist(),
        )
    )
    return sum(abs(count - edges[(j, i)]) for (i, j), count in edges.items())


@pytest.mark.parametrize("make", VESSELS)
def test_vessel_meshes_are_closed_and_small(make) -> None:
    mesh = make().mesh
    assert unpaired_edges(mesh) == 0
    assert 12 < len(mesh.triangles) < 8000
    assert mesh.colors.shape == mesh.vertices.shape
    assert int(mesh.triangles.max()) < len(mesh.vertices)


def test_solids_face_outward_with_the_right_volume() -> None:
    radius, length = 0.1, 2.0
    # A 12-sided prism has cross-section area 3 r^2.
    assert signed_volume(tube((0, 0, 0), (length, 0, 0), radius)) == pytest.approx(
        3 * radius**2 * length, rel=1e-4
    )
    # A NACA 00xx section has area 0.685 t c^2.
    foil = foil_mesh(0.1, 1.0, 0.12, 1.0, (0, 0, 1), (0.0, 1.0, 0.0), stations=2)
    assert signed_volume(foil) == pytest.approx(0.685 * 0.12 * 0.1**2 * 1.0, rel=0.03)
    assert signed_volume(pod((0, 0, 0), (0, 0, 1), 0.05)) > 0


def test_sail_is_a_two_sided_sheet() -> None:
    panel = sail((0, 0, 0), (0, 0, -4), (-2, 0, 0), camber=0.1)
    assert unpaired_edges(panel) == 0
    assert signed_volume(panel) == pytest.approx(0.0, abs=1e-6)
    assert panel.vertices[:, 1].max() > 0.1


def test_load_mesh_reads_an_obj_cube(tmp_path) -> None:
    corners = [(x, y, z) for z in (0, 1) for y in (0, 1) for x in (0, 1)]
    faces = ["1 3 4 2", "5 6 8 7", "1 2 6 5", "2 4 8 6", "4 3 7 8", "3 1 5 7"]
    lines = [f"v {x} {y} {z}" for x, y, z in corners] + [f"f {face}" for face in faces]
    (tmp_path / "cube.obj").write_text("\n".join(lines))
    cube = load_mesh(tmp_path / "cube.obj", scale=2.0, offset=(1.0, 0.0, 0.0))
    assert len(cube.triangles) == 12
    assert unpaired_edges(cube) == 0
    assert signed_volume(cube) == pytest.approx(8.0)
