import os

import numpy as np
import scipy.ndimage as ndi
from skimage import measure
import matplotlib.pyplot as plt


def _infer_format(output_path, render_format):
    if render_format:
        return render_format.lower()
    _, ext = os.path.splitext(output_path)
    if ext:
        return ext.lstrip(".").lower()
    return "ply"


def select_pore_labels(
    pore_sizes,
    voxel_size,
    min_diameter_microns,
    max_diameter_microns,
    max_pores,
):
    pore_sizes = np.asarray(pore_sizes, dtype=np.float64)
    pore_volumes = pore_sizes * (voxel_size ** 3)
    pore_diameters = (6 * pore_volumes / np.pi) ** (1.0 / 3.0)

    valid_mask = np.ones_like(pore_diameters, dtype=bool)
    if min_diameter_microns and min_diameter_microns > 0:
        valid_mask &= pore_diameters >= min_diameter_microns
    if max_diameter_microns and max_diameter_microns > 0:
        valid_mask &= pore_diameters <= max_diameter_microns

    candidate_idx = np.where(valid_mask)[0]
    if candidate_idx.size == 0:
        return np.array([], dtype=np.int64), pore_diameters, pore_volumes

    order = np.argsort(pore_volumes[candidate_idx])[::-1]
    candidate_idx = candidate_idx[order]
    if max_pores and max_pores > 0 and candidate_idx.size > max_pores:
        candidate_idx = candidate_idx[:max_pores]

    return candidate_idx + 1, pore_diameters, pore_volumes


def build_pore_mesh(labeled_pores, label_ids, voxel_size=1.0, colorize=True, padding=1):
    if len(label_ids) == 0:
        return np.empty((0, 3)), np.empty((0, 3), dtype=np.int64), None

    slices = ndi.find_objects(labeled_pores)
    vertices_list = []
    faces_list = []
    colors_list = []
    vertex_offset = 0

    cmap = plt.get_cmap("turbo")
    total = len(label_ids)

    for idx, label in enumerate(label_ids):
        slc = slices[label - 1] if label - 1 < len(slices) else None
        if slc is None:
            continue

        z0 = max(slc[0].start - padding, 0)
        y0 = max(slc[1].start - padding, 0)
        x0 = max(slc[2].start - padding, 0)
        z1 = min(slc[0].stop + padding, labeled_pores.shape[0])
        y1 = min(slc[1].stop + padding, labeled_pores.shape[1])
        x1 = min(slc[2].stop + padding, labeled_pores.shape[2])

        subvolume = (labeled_pores[z0:z1, y0:y1, x0:x1] == label).astype(np.uint8)
        if not np.any(subvolume):
            continue

        try:
            verts, faces, _, _ = measure.marching_cubes(subvolume, level=0.5)
        except ValueError:
            continue

        verts[:, 0] += z0
        verts[:, 1] += y0
        verts[:, 2] += x0
        verts = verts[:, ::-1]
        if voxel_size != 1.0:
            verts *= voxel_size

        vertices_list.append(verts)
        faces_list.append(faces + vertex_offset)
        vertex_offset += verts.shape[0]

        if colorize:
            t = 0.0 if total == 1 else idx / (total - 1)
            rgb = (np.array(cmap(t)[:3]) * 255).astype(np.uint8)
            colors_list.append(np.repeat(rgb[None, :], verts.shape[0], axis=0))

        if (idx + 1) % 10 == 0 or idx + 1 == total:
            print(f"Rendered {idx + 1}/{total} pores...")

    if not vertices_list:
        return np.empty((0, 3)), np.empty((0, 3), dtype=np.int64), None

    vertices = np.vstack(vertices_list)
    faces = np.vstack(faces_list).astype(np.int64)
    colors = np.vstack(colors_list) if colorize and colors_list else None
    return vertices, faces, colors


def _write_ply(output_path, vertices, faces, colors=None):
    with open(output_path, "w") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(vertices)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        if colors is not None:
            f.write("property uchar red\n")
            f.write("property uchar green\n")
            f.write("property uchar blue\n")
        f.write(f"element face {len(faces)}\n")
        f.write("property list uchar int vertex_indices\n")
        f.write("end_header\n")

        if colors is not None:
            for vert, color in zip(vertices, colors):
                f.write(f"{vert[0]} {vert[1]} {vert[2]} {int(color[0])} {int(color[1])} {int(color[2])}\n")
        else:
            for vert in vertices:
                f.write(f"{vert[0]} {vert[1]} {vert[2]}\n")

        for face in faces:
            f.write(f"3 {int(face[0])} {int(face[1])} {int(face[2])}\n")


def _write_obj(output_path, vertices, faces):
    with open(output_path, "w") as f:
        for vert in vertices:
            f.write(f"v {vert[0]} {vert[1]} {vert[2]}\n")
        for face in faces:
            f.write(f"f {int(face[0] + 1)} {int(face[1] + 1)} {int(face[2] + 1)}\n")


def save_mesh(output_path, vertices, faces, colors=None, render_format=None):
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    mesh_format = _infer_format(output_path, render_format)
    if mesh_format == "ply":
        _write_ply(output_path, vertices, faces, colors=colors)
    elif mesh_format == "obj":
        _write_obj(output_path, vertices, faces)
    else:
        raise ValueError(f"Unsupported mesh format: {mesh_format}")

    return mesh_format


def render_pores_to_mesh(
    labeled_pores,
    pore_sizes,
    voxel_size,
    output_path,
    render_format=None,
    min_diameter_microns=0.0,
    max_diameter_microns=None,
    max_pores=0,
):
    mesh_format = _infer_format(output_path, render_format)
    label_ids, _, _ = select_pore_labels(
        pore_sizes,
        voxel_size,
        min_diameter_microns,
        max_diameter_microns,
        max_pores,
    )
    if label_ids.size == 0:
        return 0, 0, 0, None

    print(f"Building mesh for {label_ids.size} pores...")
    vertices, faces, colors = build_pore_mesh(
        labeled_pores,
        label_ids,
        voxel_size=voxel_size,
        colorize=mesh_format == "ply",
    )

    if vertices.size == 0 or faces.size == 0:
        return 0, 0, 0, None

    mesh_format = save_mesh(output_path, vertices, faces, colors=colors, render_format=mesh_format)
    return label_ids.size, vertices.shape[0], faces.shape[0], mesh_format
