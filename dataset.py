import tqdm
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import InMemoryDataset
from torch_geometric.utils.convert import from_networkx
from torch_geometric.data import HeteroData, Data
from typing import List
import wntr


def get_link(n1: str, n2: str, water_network: wntr.network.WaterNetworkModel):
    for pipe_name, pipe in water_network.pipes():
        if pipe.start_node_name == n1 and pipe.end_node_name == n2:
            return pipe
    for pump_name, pump in water_network.pumps():
        if pump.start_node_name == n1 and pump.end_node_name == n2:
            return pump
    for valve_name, valve in water_network.valves():
        if valve.start_node_name == n1 and valve.end_node_name == n2:
            return valve


def load_edges(
        pyg_graph: Data,
        networkx_graph,
        water_network: wntr.network.WaterNetworkModel,
        pressure_nodes: List[str],
        no_pressure_nodes: List[str],
        labels: pd.DataFrame,
        node_id_mapping,
        mode
):
    dict_ID_to_Node = dict()
    i = 0
    for n in networkx_graph.nodes():
        networkx_graph.nodes[n]["ID"] = i
        dict_ID_to_Node[i] = n
        i += 1

    num_nodes = len(pressure_nodes) + len(no_pressure_nodes)
    ID_pressure = [
        id for id in range(num_nodes) if not dict_ID_to_Node[id] in no_pressure_nodes
    ]
    ID_no_pressure = [
        id for id in range(num_nodes) if not dict_ID_to_Node[id] in pressure_nodes
    ]
    pressure_to_pressure = []
    pressure_to_pressure_attr = []
    pressure_to_pressure_labels = []
    pressure_to_no_pressure = []
    pressure_to_no_pressure_attr = []
    pressure_to_no_pressure_labels = []
    no_pressure_to_pressure = []
    no_pressure_to_pressure_attr = []
    no_pressure_to_pressure_labels = []
    no_pressure_to_no_pressure = []
    no_pressure_to_no_pressure_attr = []
    no_pressure_to_no_pressure_labels = []
    pressure_to_pressure_weights = []
    pressure_to_no_pressure_weights = []
    no_pressure_to_pressure_weights = []
    no_pressure_to_no_pressure_weights = []

    # Define min and max for Normalization
    diameters = [water_network.links[pipe].diameter for pipe in water_network.links if
                 hasattr(water_network.links[pipe], "diameter")]
    dmax = max(diameters)
    dmin = min(diameters)
    lengths = [water_network.links[pipe].length for pipe in water_network.links if
               hasattr(water_network.links[pipe], "length")]
    lmax = max(lengths)
    lmin = min(lengths)
    roughnesses = [water_network.links[pipe].roughness for pipe in water_network.links if
                   hasattr(water_network.links[pipe], "roughness")]
    rmax = max(roughnesses)
    rmin = min(roughnesses)

    wn = water_network
    # Modify the water network model
    if mode == "train":
        wn.options.time.duration = 393 * 24 * 3600
    else:
        wn.options.time.duration = 365 * 24 * 3600  # 365 days
    wn.options.time.hydraulic_timestep = 60 * 5  # 5min
    wn.options.time.report_timestep = 60 * 5

    sim = wntr.sim.EpanetSimulator(wn)
    results = sim.run_sim()
    if mode == "train":
        low = 0
        high = len(labels)
    elif mode == "val":
        low = 84096
        high = 105120
    elif mode == "test":
        low = 0
        high = 105120
    flowrates = results.link["flowrate"]

    # Define edge index
    for id1, id2 in zip(pyg_graph.edge_index[0], pyg_graph.edge_index[1]):
        link = get_link(
            dict_ID_to_Node[int(id1)], dict_ID_to_Node[int(id2)], water_network
        )
        flowrate = abs(flowrates[link.name])
        diameter = link.diameter if hasattr(link, "diameter") else 0.5
        length = link.length if hasattr(link, "length") else 0.0
        roughness = link.roughness if hasattr(link, "roughness") else 0.0

        """# Calculate Headloss for each pipe using the Hazen-Williams equation
        C = 140  # Hazen-Williams roughness coefficient, The average value of “C” for new pipe in steel was found to be 144, while for the older systems, the average was found to be 140 for CCP, 150 for PVC, 155 for HDPE
        velocity = flowrate / (np.pi * (diameter / 2) ** 2)
        headloss = 10.67 * (velocity / C) ** 1.852 * length / (diameter ** 4.87)
        link_weight = headloss.values"""

        # Calculate the headloss for each pipe using the Darcy-Weisbach equation
        k = roughness  # Pipe roughness coefficient (k in mm between 0,001 for copper and 4.0 for crusty steel)
        velocity = flowrate / (np.pi * (diameter / 2) ** 2)
        friction_factor = 0.2083 * (k / diameter + 0.0001) ** 0.852
        headloss = friction_factor * (length / diameter) * (velocity ** 2) / (2 * 9.81)
        link_weight = headloss.values

        """# Calculate the headloss for each pipe using the Prony equation
        A = 0.0259  # Prony equation coefficient choosen according to p.10 w/o WF https://ntrs.nasa.gov/api/citations/20000052499/downloads/20000052499.pdf
        B = 0.0507  # Prony equation coefficient
        velocity = flowrate / (np.pi * (diameter / 2) ** 2)
        headloss = A * (velocity ** 2) + B * velocity
        link_weight = headloss.values"""

        """# Calculate the headloss for each pipe using Manning's equation
        n = 0.014
        radius = diameter / 2
        area = np.pi * radius ** 2
        velocity = flowrate / area
        # Calculate elevation difference using wntr.network.WaterNetworkModel.get_headloss() method
        #elevation_difference = wn.get_headloss(link.name) # abs(link.start.elevation - link.end.elevation)
        elevation_difference = 1 if not hasattr(link.start_node, "elevation") or not hasattr(link.end_node, "elevation") else abs(link.start_node.elevation - link.end_node.elevation)
        # Calculate slope using elevation difference
        if length == 0:
            link_weight = np.zeros(105121)
        else:
            slope = elevation_difference / length
            headloss = (1 / n ** 2) * (velocity / (radius * np.sqrt(slope))) ** (2 / 3)
            link_weight = headloss.values"""

        """# Calculate the headloss for each pipe using Hagen-Poiseuille equation
        viscosity = 1e-6  # Fluid viscosity
        radius = diameter / 2
        area = np.pi * radius ** 2
        velocity = flowrate / area
        headloss = (128 * viscosity * length * velocity) / (np.pi * diameter ** 4)
        link_weight = headloss.values"""

        # Some links are not pipes but pumps or valves and therefore don't have some attributes
        # We just set 0 for now
        # Then min-max-scale
        diameter = (diameter - dmin) / (dmax - dmin)
        length = (length - lmin) / (lmax - lmin)
        roughness = (roughness - rmin) / (rmax - rmin)

        # Link labels contains the binary leakage label for all timesteps
        link_labels = (
            list(labels[str(link)]) if str(link) in labels.columns else [False] * len(labels)
        )

        if id1 in ID_pressure and id2 in ID_pressure:
            pressure_to_pressure.append([node_id_mapping[id1][1], node_id_mapping[id2][1]])
            pressure_to_pressure_attr.append([diameter, length, roughness])
            pressure_to_pressure_labels.append(link_labels)
            pressure_to_pressure_weights.append(link_weight)
        elif id1 in ID_pressure and id2 in ID_no_pressure:
            pressure_to_no_pressure.append([node_id_mapping[id1][1], node_id_mapping[id2][1]])
            pressure_to_no_pressure_attr.append([diameter, length, roughness])
            pressure_to_no_pressure_labels.append(link_labels)
            pressure_to_no_pressure_weights.append(link_weight)
        elif id1 in ID_no_pressure and id2 in ID_pressure:
            no_pressure_to_pressure.append([node_id_mapping[id1][1], node_id_mapping[id2][1]])
            no_pressure_to_pressure_attr.append([diameter, length, roughness])
            no_pressure_to_pressure_labels.append(link_labels)
            no_pressure_to_pressure_weights.append(link_weight)
        elif id1 in ID_no_pressure and id2 in ID_no_pressure:
            no_pressure_to_no_pressure.append([node_id_mapping[id1][1], node_id_mapping[id2][1]])
            no_pressure_to_no_pressure_attr.append([diameter, length, roughness])
            no_pressure_to_no_pressure_labels.append(link_labels)
            no_pressure_to_no_pressure_weights.append(link_weight)

    weights = np.concatenate((np.stack(pressure_to_no_pressure_weights), np.stack(no_pressure_to_pressure_weights), np.stack(no_pressure_to_no_pressure_weights))).T

    pressure_to_pressure_weights = np.array(pressure_to_pressure_weights)
    pressure_to_no_pressure_weights = np.array(pressure_to_no_pressure_weights)[:, low:high]
    no_pressure_to_pressure_weights = np.array(no_pressure_to_pressure_weights)[:, low:high]
    no_pressure_to_no_pressure_weights = np.array(no_pressure_to_no_pressure_weights)[:, low:high]

    # scale weights to [0, 1]
    pressure_to_pressure_weights = (pressure_to_pressure_weights - weights.min()) / (weights.max() - weights.min())
    pressure_to_no_pressure_weights = (pressure_to_no_pressure_weights - weights.min()) / (weights.max() - weights.min())
    no_pressure_to_pressure_weights = (no_pressure_to_pressure_weights - weights.min()) / (weights.max() - weights.min())
    no_pressure_to_no_pressure_weights = (no_pressure_to_no_pressure_weights - weights.min()) / (weights.max() - weights.min())

    pressure_to_pressure = torch.tensor(pressure_to_pressure)
    pressure_to_no_pressure = torch.tensor(pressure_to_no_pressure)
    no_pressure_to_pressure = torch.tensor(no_pressure_to_pressure)
    no_pressure_to_no_pressure = torch.tensor(no_pressure_to_no_pressure)
    pressure_to_pressure_attr = torch.tensor(pressure_to_pressure_attr)
    pressure_to_no_pressure_attr = torch.tensor(pressure_to_no_pressure_attr)
    no_pressure_to_pressure_attr = torch.tensor(no_pressure_to_pressure_attr)
    no_pressure_to_no_pressure_attr = torch.tensor(no_pressure_to_no_pressure_attr)
    pressure_to_pressure_labels = torch.tensor(pressure_to_pressure_labels)
    pressure_to_no_pressure_labels = torch.tensor(pressure_to_no_pressure_labels)
    no_pressure_to_pressure_labels = torch.tensor(no_pressure_to_pressure_labels)
    no_pressure_to_no_pressure_labels = torch.tensor(no_pressure_to_no_pressure_labels)

    pressure_to_pressure_weights = torch.tensor(pressure_to_pressure_weights)
    pressure_to_no_pressure_weights = torch.tensor(pressure_to_no_pressure_weights)
    no_pressure_to_pressure_weights = torch.tensor(no_pressure_to_pressure_weights)
    no_pressure_to_no_pressure_weights = torch.tensor(no_pressure_to_no_pressure_weights)

    if len(pressure_to_pressure.shape) == 2:
        pressure_to_pressure = pressure_to_pressure.transpose(0, 1)
    if len(pressure_to_no_pressure.shape) == 2:
        pressure_to_no_pressure = pressure_to_no_pressure.transpose(0, 1)
    if len(no_pressure_to_pressure.shape) == 2:
        no_pressure_to_pressure = no_pressure_to_pressure.transpose(0, 1)
    if len(no_pressure_to_no_pressure.shape) == 2:
        no_pressure_to_no_pressure = no_pressure_to_no_pressure.transpose(0, 1)

    return (
        pressure_to_pressure,
        pressure_to_no_pressure,
        no_pressure_to_pressure,
        no_pressure_to_no_pressure,
        pressure_to_pressure_attr,
        pressure_to_no_pressure_attr,
        no_pressure_to_pressure_attr,
        no_pressure_to_no_pressure_attr,
        pressure_to_pressure_labels,
        pressure_to_no_pressure_labels,
        no_pressure_to_pressure_labels,
        no_pressure_to_no_pressure_labels,
        pressure_to_pressure_weights,
        pressure_to_no_pressure_weights,
        no_pressure_to_pressure_weights,
        no_pressure_to_no_pressure_weights
    )


def create_sample(
        pressures,
        demands_for_pnodes,
        demands_for_other_nodes,
        # demands_for_nodes,
        ptp,
        ptnp,
        nptp,
        nptnp,
        ptp_attr,
        ptnp_attr,
        nptp_attr,
        nptnp_attr,
        ptp_labels,
        ptnp_labels,
        nptp_labels,
        nptnp_labels,
        ptp_weights,
        ptnp_weights,
        nptp_weights,
        nptnp_weights
):
    sample = HeteroData()

    sample["pnode"].x = torch.stack((demands_for_pnodes, pressures), dim=-1)
    sample["onode"].x = torch.unsqueeze(demands_for_other_nodes, dim=-1)

    if len(ptp) > 0:
        sample["pnode", "connects", "pnode"].edge_index = ptp
        sample["pnode", "connects", "pnode"].edge_attr = ptp_attr
        sample["pnode", "connects", "pnode"].edge_label = ptp_labels
        sample["pnode", "connects", "pnode"].edge_weight = ptp_weights
    if len(ptnp) > 0:
        sample["pnode", "connects", "onode"].edge_index = ptnp
        sample["pnode", "connects", "onode"].edge_attr = ptnp_attr
        sample["pnode", "connects", "onode"].edge_label = ptnp_labels
        sample["pnode", "connects", "onode"].edge_weight = ptnp_weights
    if len(nptp) > 0:
        sample["onode", "connects", "pnode"].edge_index = nptp
        sample["onode", "connects", "pnode"].edge_attr = nptp_attr
        sample["onode", "connects", "pnode"].edge_label = nptp_labels
        sample["onode", "connects", "pnode"].edge_weight = nptp_weights
    if len(nptnp) > 0:
        sample["onode", "connects", "onode"].edge_index = nptnp
        sample["onode", "connects", "onode"].edge_attr = nptnp_attr
        sample["onode", "connects", "onode"].edge_label = nptnp_labels
        sample["onode", "connects", "onode"].edge_weight = nptnp_weights

    return sample


class BattleDIMDataset(InMemoryDataset):
    def __init__(self, root, mode="train", transform=None, pre_transform=None, pre_filter=None):
        self.mode = mode
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        if self.mode == "train" or self.mode == "val":
            return [
                "dataset/2018_SCADA_Pressures.csv",
                "dataset/predicted_demands_2018.zip",
                "dataset/2018_Leakages.csv",
            ]
        elif self.mode == "test":
            return [
                "dataset/2019_SCADA_Pressures.csv",
                "dataset/predicted_demands_2019.zip",
                "dataset/2019_Leakages.csv",
            ]

    @property
    def processed_file_names(self):
        if self.mode == "train":
            return ["train_data.pt"]
        elif self.mode == "val":
            return ["val_data.pt"]
        elif self.mode == "test":
            return ["test_data.pt"]

    def process(self):
        inp_file = "dataset/L-TOWN.inp"
        wn = wntr.network.WaterNetworkModel(inp_file)
        G = wn.get_graph()
        # convert to pyg graph
        pyg_graph: Data = from_networkx(G)

        df_pressures = pd.read_csv(self.raw_file_names[0], delimiter=";", decimal=",")
        df_demands = pd.read_csv(self.raw_file_names[1], compression="zip")
        df_labels = pd.read_csv(self.raw_file_names[2], delimiter=";", decimal=",")

        num_samples = len(df_pressures)
        train_share = int(0.8 * num_samples)

        if self.mode == "train":
            df_pressures = df_pressures.iloc[:train_share]
            df_demands = df_demands.iloc[:train_share]
            df_labels = df_labels.iloc[:train_share]
        elif self.mode == "val":
            df_pressures = df_pressures.iloc[train_share:]
            df_demands = df_demands.iloc[train_share:]
            df_labels = df_labels.iloc[train_share:]

        num_samples = len(df_pressures)

        # Binarize labels
        breakable_links = df_labels.columns[1:]
        df_labels[breakable_links] = df_labels[breakable_links] > 0.0

        nodes_with_pressures = df_pressures.columns[1:]
        nodes_without_pressures = [
            n for n in df_demands.columns[2:] if n not in nodes_with_pressures
        ]

        # edges are labeled globally -> need a mapping
        i_pnodes = 0
        i_onodes = 0
        node_id_mapping = []
        for i_total in df_demands.columns[2:]:
            if i_total in nodes_with_pressures:
                node_id_mapping.append(('p', i_pnodes))
                i_pnodes += 1
            else:
                node_id_mapping.append(('o', i_onodes))
                i_onodes += 1

        # Build edges
        (
            ptp,
            ptnp,
            nptp,
            nptnp,
            ptp_attr,
            ptnp_attr,
            nptp_attr,
            nptnp_attr,
            ptp_labels,
            ptnp_labels,
            nptp_labels,
            nptnp_labels,
            ptp_weights,
            ptnp_weights,
            nptp_weights,
            nptnp_weights
        ) = load_edges(
            pyg_graph, G, wn, nodes_with_pressures, nodes_without_pressures, df_labels, node_id_mapping, self.mode
        )

        data_list = []

        # pnode_pressures = df_pressures.values[:, 1:].astype(np.float32)
        # pnode_demands = df_demands[nodes_with_pressures].values.astype(np.float32)
        # onode_demands = df_demands[nodes_without_pressures].values.astype(np.float32)

        dict_base_demand = {}
        for j in wn.junctions():
            dict_base_demand[j[0]] = j[1].demand_timeseries_list.base_demand_list()
        base_demands = np.zeros((1, len(dict_base_demand)))
        for idx, i in enumerate(dict_base_demand.keys()):
            base_demands[0, idx] = max(dict_base_demand[i])
        base_demands = pd.DataFrame(base_demands, columns=list(dict_base_demand.keys()))    # base could be used instead of the Neuralprophet predictions

        # Scale node features
        pnode_pressures = df_pressures.values[:, 1:].astype(np.float32)
        ppmn, ppmx = pnode_pressures.min(), pnode_pressures.max()
        pnode_pressures = (pnode_pressures - ppmn) / (ppmx - ppmn)

        pnode_demands = df_demands[nodes_with_pressures].values.astype(np.float32)
        pdmn, pdmx = pnode_demands.min(), pnode_demands.max()
        pnode_demands = (pnode_demands - pdmn) / (pdmx - pdmn)

        onode_demands = df_demands[nodes_without_pressures].values.astype(np.float32)
        odmn, odmx = onode_demands.min(), onode_demands.max()
        onode_demands = (onode_demands - odmn) / (odmx - odmn)

        for i in tqdm.tqdm(range(num_samples)):
            sample = create_sample(
                torch.Tensor(pnode_pressures[i]),
                torch.Tensor(pnode_demands[i] % len(onode_demands)),
                torch.Tensor(onode_demands[i] % len(onode_demands)),
                ptp,
                ptnp,
                nptp,
                nptnp,
                ptp_attr,
                ptnp_attr,
                nptp_attr,
                nptnp_attr,
                torch.tensor(0),
                ptnp_labels[:, i],
                nptp_labels[:, i],
                nptnp_labels[:, i],
                torch.tensor(0),
                ptnp_weights[:, i],
                nptp_weights[:, i],
                nptnp_weights[:, i]
            )

            # Read data into huge `Data` list.
            data_list.append(sample)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


if __name__ == '__main__':
    ds = BattleDIMDataset(root="./", mode="val")
    print(ds.get(0)['pnode', 'connects', 'onode'])

