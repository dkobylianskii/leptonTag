from numpy.core.numeric import full
import uproot
import numpy as np
import pandas as pd
import pickle
from tqdm import tqdm


import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Sampler
import itertools
import math

import dgl
import dgl.function as fn
import torch.nn.functional as F
from dgl import DGLGraph


def collate_graphs(samples):
    graphs = [x[0] for x in samples]
    jet_flavor_targets = torch.cat([x[1] for x in samples])
    batched_graph = dgl.batch(graphs)

    return batched_graph, jet_flavor_targets


def EdgeLabelFunction(edges):

    edge_labels = (edges.src["node vtx idx"] > -1) & (
        edges.src["node vtx idx"] == edges.dst["node vtx idx"]
    )

    return {"edge label": edge_labels.int().float()}


class JetsDataset(Dataset):
    def __init__(self, filename, config, reduce_ds=1.0, skip_start=0.0):

        self.var_transformations = config["var_transformations"]
        self.scale_factors = config["scale_factor"]

        # with open("edge_set.pkl", "rb") as f:
        #     self.edge_lists = pickle.load(f)
        self.edge_lists = {
            N: np.array(list(itertools.permutations(range(N), 2)))
            for N in range(1, 200)
        }

        rootfile = uproot.open(filename)
        self.tree = rootfile["output_tree_el"]
        self.njets = self.tree.num_entries
        print("Number of jets: ", self.njets)
        if reduce_ds < 1.0:
            self.njets = int(self.njets * reduce_ds)

        if reduce_ds > 1.0:
            self.njets = reduce_ds
        self.start = skip_start

        self.flav_class_dict = {5: 0, 4: 1, 0: 2, 15: 3, 54: 0, 55: 0, 44: 1}

        self.jet_variables = ["jet_score", "jet_pt", "jet_e", "jet_eta", "jet_phi"]

        # self.cell_variables = ["cell_e", "cell_eta", "cell_phi"]

        self.track_variables = [
            "trk_extrap_node_d0",
            "trk_extrap_node_z0",
            "trk_extrap_node_phi0",
            "trk_extrap_node_theta",
            "trk_extrap_node_qoverp",
        ]
        self.track_common_variables = [
            "trk_extrap_node_d0",
            "trk_extrap_node_z0",
            "trk_extrap_node_phi0",
            "trk_extrap_node_theta",
            "trk_extrap_node_qoverp",
        ]

        self.lepton_common_variables = [
            "lepton_d0",
            "lepton_z0",
            "lepton_track_phi",
            "lepton_track_theta",
            "lepton_track_qoverp",
        ]
        self.lepton_variables = [
            "lepton_d0",
            "lepton_z0",
            "lepton_track_phi",
            "lepton_track_theta",
            "lepton_track_qoverp",
            "lepton_ptRel",
            "lepton_e",
            "lepton_phi",
            "lepton_eta",
            "lepton_f1",
            "lepton_f3",
            "lepton_f3core",
            "lepton_weta1",
            "lepton_weta2",
            "lepton_fracs1",
            "lepton_wtots1",
            "lepton_e277",
            "lepton_Reta",
            "lepton_Rphi",
            "lepton_Eratio",
            "lepton_Rhad",
            "lepton_Rhad1",
            "lepton_deltaEta1",
            "lepton_deltaPhi1",
            "lepton_deltaPhi2",
        ]

        self.extra_vars = [
            "jet_ntracks",
            "trk_vertex_index",
            "lepton_vertex_index",
            "reco_type",
            "PVz",
        ]

        self.cell_variables = ["cell_e", "cell_eta", "cell_phi"]

        self.variables = (
            self.jet_variables
            + self.track_variables
            + self.lepton_variables
            + self.extra_vars
            + self.cell_variables
            + ["cell_r"]
        )

        self.jet_flavs = self.tree["target_jet_type"].array(
            library="np", entry_start=self.start, entry_stop=self.njets
        )

        self.full_data_array = {}
        print("loading all variables")
        for var in tqdm(self.variables):
            # print(var)
            self.full_data_array[var] = self.tree[var].array(
                library="np", entry_start=self.start, entry_stop=self.njets
            )
            if var in self.track_variables + self.lepton_variables:
                self.full_data_array[var] = np.concatenate(self.full_data_array[var])
            if var in self.cell_variables:
                self.full_data_array[var] = np.concatenate(self.full_data_array[var])
        print("done")

        self.full_n_tracks = self.full_data_array["jet_ntracks"]

        self.full_n_leptons = np.array(
            [len(x) for x in self.full_data_array["reco_type"]]
        )

        self.full_n_cells = np.array([len(x) for x in self.full_data_array["cell_r"]])
        self.cell_variables += ["cell_r"]
        self.full_data_array["cell_r"] = np.concatenate([x for x in self.full_data_array["cell_r"]])

        pvz_for_leptons = np.repeat(self.full_data_array["PVz"], self.full_n_leptons)
        self.full_data_array["lepton_z0"] = (
            self.full_data_array["lepton_z0"] - pvz_for_leptons
        )

        # transform the variables
        for var in self.scale_factors:
            value_arr = self.full_data_array[var]

            self.full_data_array[var] = np.sign(value_arr) * np.log(
                np.abs(value_arr * self.scale_factors[var]) + 1
            )

        # scale the variables
        for var in self.var_transformations:
            self.full_data_array[var] = (
                self.full_data_array[var] - self.var_transformations[var]["mean"]
            ) / self.var_transformations[var]["std"]

        trk_from_B = np.concatenate(
            self.tree["trk_from_B"].array(
                library="np", entry_start=self.start, entry_stop=self.njets
            )
        )
        trk_from_C = np.concatenate(
            self.tree["trk_from_C"].array(
                library="np", entry_start=self.start, entry_stop=self.njets
            )
        )
        self.track_node_labels = self.create_class_labels(
            np.concatenate(self.full_data_array["trk_vertex_index"]),
            trk_from_B,
            trk_from_C,
        )
        self.track_node_labels = np.split(
            self.track_node_labels, np.cumsum(self.full_n_tracks)
        )
        reco_type = [x for x in self.full_data_array["reco_type"]]
        self.lepton_node_labels = np.concatenate(reco_type)
        self.lepton_node_labels = np.split(
            self.lepton_node_labels, np.cumsum(self.full_n_leptons)
        )

        # 0 for track, 1 for electron, 2 for muon, 3 for calo cell
        self.node_type_dict = {0: 0, 11: 1, 13: 2}


        self.track_node_type = np.zeros(len(self.full_data_array["trk_extrap_node_d0"]))
        self.track_node_type = np.split(
            self.track_node_type, np.cumsum(self.full_n_tracks)
        )
        self.track_node_type = [torch.LongTensor(x) for x in self.track_node_type]

        leptontype = 11 * np.ones_like(
            np.concatenate(self.full_data_array["reco_type"])
        )
        leptontype = np.array([self.node_type_dict[x] for x in leptontype])
        self.lepton_node_type = np.split(leptontype, np.cumsum(self.full_n_leptons))
        self.lepton_node_type = [torch.LongTensor(x) for x in self.lepton_node_type]

        for var in self.track_variables + self.lepton_variables + self.cell_variables:
            if var in self.cell_variables:
                self.full_data_array[var] = np.split(
                    self.full_data_array[var], np.cumsum(self.full_n_cells)
                )
            if var in self.track_variables:
                self.full_data_array[var] = np.split(
                    self.full_data_array[var], np.cumsum(self.full_n_tracks)
                )
            if var in self.lepton_variables:
                self.full_data_array[var] = np.split(
                    self.full_data_array[var], np.cumsum(self.full_n_leptons)
                )

    def get_single_jet(self, idx):

        n_tracks = self.full_n_tracks[idx]
        n_leptons = self.full_n_leptons[idx]
        

        # translate from flavor label 5,4,0 to 0,1,2
        jet_flav = torch.LongTensor([self.jet_flavs[idx]])

        # jet_flav = torch.LongTensor([self.flav_class_dict[jet_flav]])

        full_array = {
            var: self.full_data_array[var][idx]
            for var in self.jet_variables
            + self.track_variables
            + self.lepton_variables
            + self.cell_variables
        }
        mask = full_array["cell_e"] > 500
        for var in self.cell_variables:
            full_array[var] = full_array[var][mask]
        full_array["cell_e"] = np.log(full_array["cell_e"])
        n_cells = np.sum(mask)
        # if n_cells > 80:
        #     n_cells = 80
        #     for var in self.cell_variables:
        #         full_array[var] = full_array[var][:n_cells]
        n_nodes = n_tracks + n_leptons + n_cells

        # for var in ["_phi", "_eta"]:
        #     full_array["cell" + var] -= full_array["jet" + var]
        #     full_array["lepton" + var] -= full_array["jet" + var]

        # repeat the jet variables for tracks and leptons
        for var in self.jet_variables:
            full_array[var + "_tracks"] = np.repeat(full_array[var], n_tracks)
            full_array[var + "_leptons"] = np.repeat(full_array[var], n_leptons)
            full_array[var + "_cells"] = np.repeat(full_array[var], n_cells)

        track_vertex_idx = self.full_data_array["trk_vertex_index"][idx]
        lepton_vertex_idx = self.full_data_array["lepton_vertex_index"][idx]

        node_vertex_idx = np.concatenate([track_vertex_idx, lepton_vertex_idx])

        vtx_set = torch.tensor(list(set(node_vertex_idx[node_vertex_idx > -1]))).int()
        n_objects = len(vtx_set)

        track_class_labels = self.track_node_labels[idx]
        lepton_class_labels = self.lepton_node_labels[idx]

        track_vertex_idx = torch.FloatTensor(track_vertex_idx)
        lepton_vertex_idx = torch.FloatTensor(lepton_vertex_idx)

        for var in full_array:
            if var in self.jet_variables:
                full_array[var] = torch.FloatTensor([full_array[var]])
            full_array[var] = torch.FloatTensor(full_array[var])
        track_class_labels = torch.FloatTensor(track_class_labels)
        lepton_class_labels = torch.FloatTensor(lepton_class_labels)

        return (
            jet_flav,
            n_nodes,
            n_tracks,
            n_leptons,
            n_cells,
            full_array,
            track_class_labels,
            lepton_class_labels,
            vtx_set,
            n_objects,
            track_vertex_idx,
            lepton_vertex_idx,
        )

    def create_class_labels(self, node_vtx_idx, from_B, from_C):
        node_labels = np.zeros(len(node_vtx_idx))  # 0 = pile up
        node_labels[node_vtx_idx == 0] = 1  # primary vertex
        node_labels[(node_vtx_idx > 0) & (from_B == 1)] = 2  # from B
        node_labels[(node_vtx_idx > 0) & (from_C == 1)] = 3  # from C
        node_labels[
            (node_vtx_idx > 0) & (from_B != 1) & (from_C != 1)
        ] = 4  # from secondary other than B/C

        return node_labels

    def __len__(self):

        return int(self.njets - self.start)

    def build_node_variables(self, data_array, variables):
        node_arr = []
        for var in variables:

            node_arr.append(data_array[var])

        return torch.stack(node_arr, dim=1)

    def load_single_graph(self, idx):

        properties = self.get_single_jet(idx)
        (
            jet_flav,
            N,
            N_track,
            N_lep,
            N_cells,
            full_array,
            track_class_labels,
            lepton_class_labels,
            vertex_set,
            n_objects,
            track_vertex_idx,
            lepton_vertex_idx,
        ) = properties
        

        if N > 1:
            if N_cells > 0:
                cell_edgelist1 = np.repeat(np.arange(N_lep) + N_track, N_cells)
                cell_edgelist2 = np.tile(torch.arange(N_cells) + N_lep + N_track, N_lep)

                node_edgelist1 = np.concatenate([
                    self.edge_lists[N - N_cells][:, 0],
                    cell_edgelist1,
                    cell_edgelist2
                ])
                node_edgelist2 = np.concatenate([
                    self.edge_lists[N - N_cells][:, 1],
                    cell_edgelist2,
                    cell_edgelist1
                ])
            else:
                node_edgelist1 = self.edge_lists[N][:, 0]
                node_edgelist2 = self.edge_lists[N][:, 1]
        else:
            node_edgelist1, node_edgelist2 = [], []

        edge_list1 = torch.repeat_interleave(torch.arange(N), n_objects)
        edge_list2 = torch.arange(n_objects).repeat(N)

        track_to_node_e = torch.arange(N_track)
        lepton_to_node_e = torch.arange(N_lep)
        cell_to_node_e = torch.arange(N_cells)

        num_nodes_dict = {
            "tracks": N_track,
            "leptons": N_lep,
            "cells": N_cells,
            "nodes": N,
            "objects": n_objects,
            "global node": 1,
        }

        data_dict = {
            ("tracks", "track_to_node", "nodes"): (track_to_node_e, track_to_node_e),
            ("leptons", "leptons_to_node", "nodes"): (
                lepton_to_node_e,
                N_track + lepton_to_node_e,
            ),
            ("cells", "cells_to_node", "nodes"): (
                cell_to_node_e,
                N_track + N_lep + cell_to_node_e,
            ),
            ("nodes", "node_to_node", "nodes"): (node_edgelist1, node_edgelist2),
            ("nodes", "to_parent", "objects"): (edge_list1, edge_list2),
            ("objects", "to_child", "nodes"): (edge_list2, edge_list1),
            ("nodes", "to_global", "global node"): (
                torch.arange(N).int(),
                torch.zeros(N).int(),
            ),
        }

        g = dgl.heterograph(data_dict, num_nodes_dict)

        if len(vertex_set) == 0:
            g.add_nodes(torch.tensor(1), ntype="objects")
            vertex_set = torch.tensor([-2]).int()
        g.nodes["objects"].data["vertex ids"] = vertex_set

        g.nodes["global node"].data["jet features"] = self.build_node_variables(
            full_array, self.jet_variables
        )

        g.nodes["tracks"].data["track variables"] = self.build_node_variables(
            full_array,
            [x + "_tracks" for x in self.jet_variables] + self.track_variables,
        )
        g.nodes["leptons"].data["lep variables"] = self.build_node_variables(
            full_array,
            [x + "_leptons" for x in self.jet_variables] + self.lepton_variables,
        )
        g.nodes["cells"].data["cell variables"] = self.build_node_variables(
            full_array, [x + "_cells" for x in self.jet_variables] + self.cell_variables
        )

        g.nodes["tracks"].data["common variables"] = self.build_node_variables(
            full_array,
            [x + "_tracks" for x in self.jet_variables] + self.track_common_variables,
        )
        g.nodes["leptons"].data["common variables"] = self.build_node_variables(
            full_array,
            [x + "_leptons" for x in self.jet_variables] + self.lepton_common_variables,
        )

        g.nodes["tracks"].data["node labels"] = track_class_labels
        g.nodes["leptons"].data["node labels"] = lepton_class_labels
        # Fill cell labels by zeros
        g.nodes["cells"].data["node labels"] = -1 * torch.ones(N_cells, dtype=torch.float32)

        g.nodes["tracks"].data["node vtx idx"] = track_vertex_idx
        g.nodes["leptons"].data["node vtx idx"] = lepton_vertex_idx
        # Fill node vtx idx by -1
        g.nodes["cells"].data["node vtx idx"] = -1 * torch.ones(N_cells, dtype=torch.float32)

        g.nodes["tracks"].data["node_type"] = self.track_node_type[idx]
        g.nodes["leptons"].data["node_type"] = self.lepton_node_type[idx]
        # Introduce cell types: 3 for em calo, 4 for hadron calo
        cell_types = 3 * (full_array['cell_r'] <= 1800) + 4 * (full_array['cell_r'] > 1800)
        g.nodes["cells"].data["node_type"] = torch.LongTensor(cell_types)
        # g.nodes['nodes'].data['node_type'] = torch.cat( [self.track_node_type[idx],self.lepton_node_type[idx]],dim=0).unsqueeze(1)
        
        return g, jet_flav

    def __getitem__(self, idx):
        return self.load_single_graph(idx)
