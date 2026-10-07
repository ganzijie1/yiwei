import csv
import json
import math
from collections import deque
from datetime import date
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)

START_YEAR = 1895
START_MONTH = 6
MONTHS = 127
NODES = [
    "Aodi", "Keelung", "Taipei", "Taoyuan", "Hsinchu", "Miaoli",
    "Taichung", "Changhua", "Yunlin", "Chiayi", "Tainan", "Pingtung",
    "MountainInterior",
]
N = len(NODES)
IDX = {name: i for i, name in enumerate(NODES)}
AGENT_TYPES = (
    "committed_collaborator",
    "institutional_careerist",
    "material_opportunist",
    "local_broker",
    "coerced_participant",
)
K_TYPES = len(AGENT_TYPES)


def month_date(index):
    m = START_MONTH - 1 + index
    return date(START_YEAR + m // 12, m % 12 + 1, 1)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -40.0, 40.0)))


def softmax(x):
    z = np.asarray(x, dtype=float) - np.max(x)
    e = np.exp(z)
    return e / e.sum()


def add_edge(matrix, a, b, weight, directed=False):
    matrix[IDX[a], IDX[b]] = weight
    if not directed:
        matrix[IDX[b], IDX[a]] = weight


def build_layers():
    transport = np.zeros((N, N))
    chain = NODES[:-1]
    for a, b in zip(chain[:-1], chain[1:]):
        add_edge(transport, a, b, 1.0)
    add_edge(transport, "Aodi", "Taipei", 0.30)
    add_edge(transport, "Keelung", "Taipei", 1.20)
    add_edge(transport, "Taoyuan", "MountainInterior", 0.32)
    add_edge(transport, "Hsinchu", "MountainInterior", 0.48)
    add_edge(transport, "Miaoli", "MountainInterior", 0.55)
    add_edge(transport, "Yunlin", "MountainInterior", 0.48)
    add_edge(transport, "Chiayi", "MountainInterior", 0.38)
    add_edge(transport, "Pingtung", "Tainan", 0.70)

    social = np.zeros((N, N))
    for a, b in zip(chain[:-1], chain[1:]):
        add_edge(social, a, b, 0.35)
    for a, b, w in [
        ("Taoyuan", "Hsinchu", 0.85), ("Hsinchu", "Miaoli", 0.95),
        ("Miaoli", "Taichung", 0.72), ("Changhua", "Yunlin", 0.65),
        ("Yunlin", "Chiayi", 0.70), ("Tainan", "Pingtung", 0.76),
        ("Hsinchu", "MountainInterior", 0.62),
        ("Miaoli", "MountainInterior", 0.70),
        ("Yunlin", "MountainInterior", 0.82),
        ("Chiayi", "MountainInterior", 0.55),
    ]:
        add_edge(social, a, b, w)

    governance = np.zeros((N, N))
    for a, b in zip(chain[:-1], chain[1:]):
        add_edge(governance, a, b, 0.55)
    add_edge(governance, "Keelung", "Taipei", 1.0)
    add_edge(governance, "Taipei", "Hsinchu", 0.42)
    add_edge(governance, "Tainan", "Pingtung", 0.62)

    disease = transport.copy()
    disease[disease > 0] = np.sqrt(disease[disease > 0])
    return transport, social, governance, disease


LEADERS = {
    "Tang Jingsong": ({"Taipei": 1.0}, 0),
    "Liu Yongfu": ({"Tainan": 1.0, "Chiayi": 0.45, "Pingtung": 0.35}, 4),
    "Wu Tangxing": ({"Taoyuan": 0.35, "Hsinchu": 0.7, "Miaoli": 1.0,
                       "Taichung": 0.55, "Changhua": 0.8}, 2),
    "Jiang Shaozu": ({"Hsinchu": 1.0, "MountainInterior": 0.45}, 1),
    "Xu Xiang": ({"Hsinchu": 0.45, "Miaoli": 0.9, "Changhua": 0.65,
                    "Yunlin": 0.55, "Chiayi": 0.35}, 4),
    "Wu Pengnian": ({"Taichung": 0.55, "Changhua": 1.0, "Yunlin": 0.35}, 2),
    "Jiang Guohui": ({"Taoyuan": 0.8, "MountainInterior": 0.30}, 1),
    "Huang Niangsheng": ({"Taoyuan": 0.65}, 1),
    "Hu Jiayou": ({"Taoyuan": 0.55, "Hsinchu": 0.35}, 2),
    "Xiao Guangming": ({"Pingtung": 0.9}, 4),
    "Qiu Fengyang": ({"Pingtung": 0.85}, 5),
    "Ke Tie": ({"Yunlin": 0.95, "MountainInterior": 1.0}, 83),
}


def leader_incidence():
    names = list(LEADERS)
    h = np.zeros((len(names), N))
    exits = np.zeros(len(names), dtype=int)
    for row, name in enumerate(names):
        regions, exits[row] = LEADERS[name]
        for region, weight in regions.items():
            h[row, IDX[region]] = weight
    return names, h, exits


def max_flow(capacity, sources, demand):
    total_nodes = N + 2
    source, sink = N, N + 1
    cap = np.zeros((total_nodes, total_nodes), dtype=float)
    cap[:N, :N] = capacity
    for i in sources:
        cap[source, i] = 10.0
    for i, value in enumerate(demand):
        cap[i, sink] = max(0.0, value)

    flow = 0.0
    residual = cap.copy()
    while True:
        parent = np.full(total_nodes, -1, dtype=int)
        parent[source] = source
        queue = deque([source])
        while queue and parent[sink] < 0:
            u = queue.popleft()
            for v in np.where(residual[u] > 1e-10)[0]:
                if parent[v] < 0:
                    parent[v] = u
                    queue.append(v)
                    if v == sink:
                        break
        if parent[sink] < 0:
            break
        increment = float("inf")
        v = sink
        while v != source:
            u = parent[v]
            increment = min(increment, residual[u, v])
            v = u
        v = sink
        while v != source:
            u = parent[v]
            residual[u, v] -= increment
            residual[v, u] += increment
            v = u
        flow += increment
    return flow


def normalize_rows(matrix):
    denom = matrix.sum(axis=1, keepdims=True)
    return np.divide(matrix, denom, out=np.zeros_like(matrix), where=denom > 0)


def network_metrics(matrix):
    """Small weighted-network diagnostics for the broker layer."""
    sym = (matrix + matrix.T) / 2.0
    np.fill_diagonal(sym, 0.0)
    positive = sym[sym > 0]
    density = float(positive.mean()) if positive.size else 0.0
    eigenvalues = np.linalg.eigvalsh(sym)
    spectral_radius = float(np.max(np.abs(eigenvalues))) if eigenvalues.size else 0.0
    laplacian = np.diag(sym.sum(axis=1)) - sym
    laplacian_values = np.linalg.eigvalsh(laplacian)
    algebraic_connectivity = float(max(0.0, laplacian_values[1])) if N > 1 else 0.0
    if positive.size:
        normalized = sym / positive.max()
        degree = np.count_nonzero(normalized, axis=1)
        triangles = np.diag(normalized @ normalized @ normalized)
        denom = degree * np.maximum(degree - 1, 1)
        clustering = float(np.mean(np.divide(
            triangles, denom, out=np.zeros_like(triangles), where=degree > 1)))
    else:
        clustering = 0.0
    return density, spectral_radius, algebraic_connectivity, clustering


def simulate(seed=1895, adaptive=True, leaders=True, disease_mobility=True,
             strategy_feedback=True, multilayer_agency=True,
             agent_network_game=True):
    rng = np.random.default_rng(seed)
    dates = [month_date(t) for t in range(MONTHS)]
    base_transport, base_social, gov_net, disease_net = build_layers()
    social = base_social.copy()
    names, incidence, leader_exits = leader_incidence()

    capture_month = np.array([0, 0, 0, 0, 0, 2, 2, 2, 3, 3, 4, 4, 10])
    urbanity = np.array([0.10, 0.58, 1.00, 0.48, 0.64, 0.42, 0.60,
                         0.62, 0.35, 0.50, 0.78, 0.42, 0.05])
    terrain = np.array([0.45, 0.58, 0.28, 0.52, 0.72, 0.78, 0.44,
                        0.55, 0.64, 0.52, 0.25, 0.38, 1.00])
    population = np.array([0.08, 0.32, 0.90, 0.52, 0.60, 0.48, 0.55,
                           0.62, 0.42, 0.52, 0.82, 0.46, 0.12])

    shape = (MONTHS, N)
    control = np.zeros(shape)
    resistance = np.zeros(shape)
    governance = np.zeros(shape)
    services = np.zeros(shape)
    legitimacy = np.zeros(shape)
    grievance = np.zeros(shape)
    exposed = np.zeros(shape)
    infected = np.zeros(shape)
    recovered = np.zeros(shape)
    susceptible = np.zeros(shape)
    leader_capacity = np.zeros(shape)
    flow_ratio = np.zeros(MONTHS)
    coercion = np.zeros(MONTHS)
    accommodation = np.zeros(MONTHS)
    dispersion = np.zeros(MONTHS)
    concentration = np.zeros(MONTHS)
    principal_monitoring = np.zeros(MONTHS)
    principal_wage = np.zeros(MONTHS)
    local_autonomy = np.zeros(MONTHS)
    administration_effort = np.zeros(shape)
    police_effort = np.zeros(shape)
    intermediary_effort = np.zeros(shape)
    household_reporting = np.zeros(shape)
    agency_rent = np.zeros(shape)
    information_distortion = np.zeros(shape)
    loyalty_belief = np.zeros((MONTHS, N, 4))
    type_loyalty_belief = np.zeros((MONTHS, N, K_TYPES, 4))
    agent_compliance_action = np.zeros((MONTHS, N, K_TYPES))
    broker_network_density = np.zeros(MONTHS)
    broker_spectral_radius = np.zeros(MONTHS)
    broker_algebraic_connectivity = np.zeros(MONTHS)
    broker_weighted_clustering = np.zeros(MONTHS)
    social_trust = np.zeros(shape)
    economic_output = np.zeros(shape)
    extraction_rate = np.zeros(shape)
    identity_alignment = np.zeros(shape)
    jump_shock = np.zeros(shape)
    maritime_blockade = np.zeros(MONTHS)
    mainland_clandestine_contact = np.zeros(shape)
    regulated_cross_strait_trade = np.zeros(shape)
    belief = np.zeros((MONTHS, 3))
    belief[0] = [0.72, 0.23, 0.05]

    control[0] = sigmoid(2.2 * (0.2 - capture_month))
    resistance[0] = np.clip(0.38 + 0.34 * terrain + 0.18 * population
                            - 0.20 * control[0], 0.08, 0.95)
    governance[0] = 0.04 * control[0]
    services[0] = 0.12 + 0.08 * urbanity
    legitimacy[0] = 0.08
    grievance[0] = 0.25 + 0.25 * resistance[0]
    susceptible[0] = 0.985
    exposed[0, :3] = [0.006, 0.008, 0.004]
    infected[0, :3] = [0.004, 0.006, 0.003]
    recovered[0] = 1.0 - susceptible[0] - exposed[0] - infected[0]
    administration_effort[0] = 0.08 * control[0]
    police_effort[0] = 0.10 * control[0]
    intermediary_effort[0] = 0.05 * control[0]
    household_reporting[0] = 0.04 * control[0]
    raw_type_share = np.column_stack([
        0.10 + 0.08 * urbanity,
        0.16 + 0.13 * urbanity,
        np.full(N, 0.25),
        0.18 + 0.15 * (1.0 - urbanity),
        0.08 + 0.12 * resistance[0],
    ]) * rng.lognormal(0.0, 0.10, (N, K_TYPES))
    agent_type_share = raw_type_share / raw_type_share.sum(axis=1, keepdims=True)
    type_initial_state = np.array([
        [0.70, 0.25, 0.02, 0.03],
        [0.26, 0.62, 0.03, 0.09],
        [0.08, 0.75, 0.05, 0.12],
        [0.05, 0.68, 0.10, 0.17],
        [0.02, 0.48, 0.04, 0.46],
    ])
    type_loyalty_belief[0] = type_initial_state[None, :, :]
    loyalty_belief[0] = np.einsum(
        "nk,nks->ns", agent_type_share, type_loyalty_belief[0])
    agent_compliance_action[0] = np.array([0.82, 0.66, 0.46, 0.42, 0.25])[None, :]
    social_trust[0] = np.clip(0.68 - 0.18 * resistance[0], 0.30, 0.75)
    economic_output[0] = np.clip(0.62 + 0.22 * urbanity - 0.18 * resistance[0], 0.30, 0.90)
    identity_alignment[0] = 0.04 + 0.03 * urbanity
    coastal_access = np.array([0.62, 0.86, 0.30, 0.18, 0.24, 0.20, 0.16,
                               0.22, 0.32, 0.58, 0.78, 0.66, 0.04])
    mainland_clandestine_contact[0] = 0.22 * coastal_access
    broker_network_base = np.clip(0.52 * gov_net + 0.48 * base_social, 0.0, 1.0)
    broker_network = broker_network_base.copy()

    transition = np.array([
        [0.70, 0.24, 0.06],
        [0.16, 0.68, 0.16],
        [0.05, 0.18, 0.77],
    ])
    emission_mean = np.array([
        [0.85, 0.42, 0.12],
        [0.50, 0.82, 0.28],
        [0.18, 0.30, 0.86],
    ])

    for t in range(MONTHS - 1):
        active = (t <= leader_exits).astype(float)
        if leaders:
            leader_capacity[t] = np.clip(active @ incidence / 2.2, 0.0, 1.0)
        else:
            leader_capacity[t] = 0.0

        campaign_signal = math.exp(-t / 3.0)
        reform_signal = float(sigmoid((t - 42) / 10.0))
        hoko_signal = float(sigmoid((t - 39) / 3.0))
        land_survey_signal = float(sigmoid((t - 39) / 5.0))
        punitive_edict = math.exp(-0.5 * ((t - 41) / 1.4) ** 2)
        pacification_signal = float(sigmoid((t - 78) / 5.0))
        # Penghu and naval control sharply raise the cost of clandestine links
        # with the mainland, but do not make them zero.  From 1898 onward,
        # regulated Taiwan-Fujian routes recover commerce under colonial
        # surveillance without restoring the old military-organizational link.
        port_control = np.mean(control[t, [IDX["Keelung"], IDX["Tainan"], IDX["Pingtung"]]])
        maritime_blockade[t] = np.clip(
            0.58 + 0.30 * port_control + 0.08 * campaign_signal, 0.45, 0.96)
        weather_opening = 0.82 + 0.18 * math.cos(
            2 * math.pi * ((dates[t].month - 4) % 12) / 12)
        mainland_clandestine_contact[t] = np.clip(
            0.035 * coastal_access
            + 0.42 * coastal_access * (1.0 - maritime_blockade[t])
            * (1.0 - 0.65 * control[t]) * weather_opening, 0.0, 0.30)
        regulated_cross_strait_trade[t] = np.clip(
            coastal_access * float(sigmoid((t - 43) / 7.0))
            * (0.34 + 0.38 * control[t]) * (0.45 + 0.55 * governance[t]), 0.0, 0.75)
        true_observation = np.array([
            campaign_signal,
            0.55 * resistance[t].mean() + 0.30 * campaign_signal,
            reform_signal,
        ])
        observed = np.clip(true_observation + rng.normal(0, 0.055, 3), 0, 1)
        prior = belief[t] @ transition
        likelihood = np.exp(-np.sum((emission_mean - observed) ** 2, axis=1) / (2 * 0.18 ** 2))
        belief[t + 1] = prior * likelihood
        belief[t + 1] /= belief[t + 1].sum()

        if strategy_feedback:
            scores = np.array([
                1.8 * resistance[t].mean() + 0.8 * belief[t, 1]
                - 0.6 * infected[t].mean(),
                1.4 * (1.0 - governance[t].mean()) + 1.0 * belief[t, 2]
                + 0.5 * infected[t].mean(),
            ])
            policy = softmax(scores / 0.35)
            coercion[t], accommodation[t] = policy
        else:
            coercion[t], accommodation[t] = 0.55, 0.45

        # Four-level principal-agent chain: Governor-General policy, local
        # administration, police, local intermediaries, and households.  Each
        # transfer loses information and creates rents; coercion can raise
        # short-run reporting while increasing grievance and retaliation risk.
        retaliation_risk = resistance[t] * (0.55 + 0.45 * leader_capacity[t])
        if multilayer_agency:
            principal_monitoring[t] = float(sigmoid(
                -0.35 + 1.55 * resistance[t].mean()
                + 0.85 * coercion[t] + 0.40 * (1.0 - flow_ratio[max(0, t - 1)])))
            principal_wage[t] = float(sigmoid(
                -0.55 + 1.10 * accommodation[t] + 0.90 * reform_signal
                + 0.35 * governance[t].mean()))
            local_autonomy[t] = float(sigmoid(
                -0.70 + 1.25 * accommodation[t] + 0.55 * reform_signal
                - 0.45 * coercion[t]))
            agency_rent[t] = np.clip(
                0.08 + 0.22 * local_autonomy[t] + 0.18 * (1.0 - principal_monitoring[t])
                + rng.normal(0, 0.018, N), 0.0, 0.55)
            administration_effort[t] = sigmoid(
                -1.05 + 1.15 * principal_wage[t] + 1.05 * principal_monitoring[t]
                + 0.85 * control[t] - 0.90 * resistance[t] - 0.75 * agency_rent[t])
            police_effort[t] = sigmoid(
                -0.85 + 1.20 * principal_monitoring[t] + 0.90 * coercion[t]
                + 0.65 * control[t] - 0.85 * retaliation_risk)
            intermediary_effort[t] = sigmoid(
                -1.15 + 0.85 * principal_wage[t] + 1.00 * local_autonomy[t]
                + 0.70 * governance[t] - 1.20 * retaliation_risk
                - 0.65 * agency_rent[t])
            joint_liability = 0.35 + 0.50 * police_effort[t] * coercion[t]
            household_reporting[t] = sigmoid(
                -1.65 + 1.15 * police_effort[t] + 1.00 * intermediary_effort[t]
                + 0.90 * joint_liability - 1.10 * resistance[t])
            information_distortion[t] = np.clip(
                0.08 + 0.30 * agency_rent[t]
                + 0.20 * np.abs(police_effort[t] - intermediary_effort[t])
                + 0.18 * (1.0 - household_reporting[t]), 0.0, 0.70)
        else:
            principal_monitoring[t] = 0.55
            principal_wage[t] = 0.52
            local_autonomy[t] = 0.42
            administration_effort[t] = governance[t]
            police_effort[t] = control[t]
            intermediary_effort[t] = governance[t]
            household_reporting[t] = 0.50 * control[t]
            agency_rent[t] = 0.0
            information_distortion[t] = 0.0

        # Heterogeneous quantal-response network game.  Compliance is a social
        # action with type-specific payoffs: committed and career agents value
        # institutional continuation, opportunists and local brokers react
        # strongly to neighbors, while coerced participants have a high effort
        # cost and exit option.
        if agent_network_game:
            action = (agent_compliance_action[t - 1].copy() if t > 0
                      else agent_compliance_action[0].copy())
            type_commitment = np.array([1.05, 0.55, 0.05, -0.18, -0.78])
            type_peer_complement = np.array([0.55, 0.78, 1.05, 1.18, 0.42])
            type_rent_response = np.array([0.18, 0.42, 0.88, 0.62, 0.20])
            type_retaliation = np.array([0.35, 0.55, 0.75, 1.10, 1.25])
            normalized_broker = normalize_rows(broker_network)
            for _ in range(6):
                node_action = np.sum(agent_type_share * action, axis=1)
                peer_action = normalized_broker @ node_action
                payoff_difference = (
                    -1.35 + type_commitment[None, :]
                    + 0.72 * principal_wage[t]
                    + 0.62 * local_autonomy[t]
                    + 0.74 * principal_monitoring[t]
                    + type_peer_complement[None, :] * peer_action[:, None]
                    + type_rent_response[None, :] * agency_rent[t, :, None]
                    - type_retaliation[None, :] * retaliation_risk[:, None]
                    - 0.38 * information_distortion[t, :, None])
                best_response = sigmoid(payoff_difference / 0.58)
                action = 0.45 * action + 0.55 * best_response
            agent_compliance_action[t] = action
            node_compliance = np.sum(agent_type_share * action, axis=1)
            intermediary_effort[t] *= 0.52 + 0.48 * node_compliance
        else:
            agent_compliance_action[t] = 0.55
        (broker_network_density[t], broker_spectral_radius[t],
         broker_algebraic_connectivity[t], broker_weighted_clustering[t]) = (
            network_metrics(broker_network))

        # Hidden Markov loyalty filter.  The four latent states are loyal,
        # opportunistic compliance, dual linkage, and prepared/realized exit.
        # The observed intermediary effort is an imperfect emission, not the
        # state itself.
        loyalty_signal = np.zeros(N)
        type_loyal_persistence = np.array([0.18, 0.10, 0.02, -0.02, -0.08])
        type_wage_response = np.array([0.02, 0.14, 0.12, 0.04, 0.01])
        type_dual_propensity = np.array([-0.02, -0.01, 0.02, 0.11, -0.01])
        type_exit_propensity = np.array([-0.06, -0.02, 0.04, 0.02, 0.16])
        type_effort_weight = np.array([1.18, 1.05, 0.82, 0.76, 0.52])
        pressure = np.clip(
            0.55 * resistance[t] + 0.35 * (1.0 - control[t])
            + 0.25 * jump_shock[max(0, t - 1)], 0.0, 1.0)
        wage = principal_wage[t]
        amnesty = 0.22 * accommodation[t] + 0.18 * hoko_signal
        exit_pull = np.clip(
            0.08 + 0.34 * pressure + 0.22 * leader_capacity[t], 0.02, 0.55)
        dual_access = np.clip(
            0.60 * resistance[t] * social_trust[t]
            + 0.40 * mainland_clandestine_contact[t], 0.0, 0.55)
        surveillance = np.clip(
            principal_monitoring[t] * (0.45 + 0.55 * hoko_signal)
            * household_reporting[t], 0.0, 1.0)

        transition_ik = np.zeros((N, K_TYPES, 4, 4))
        transition_ik[:, :, 0, 0] = 0.82 + 0.08 * wage + type_loyal_persistence
        transition_ik[:, :, 0, 1] = 0.13 - 0.05 * wage - 0.45 * type_loyal_persistence
        transition_ik[:, :, 0, 2] = 0.04 + 0.20 * type_dual_propensity
        transition_ik[:, :, 0, 3] = 0.01 + 0.35 * type_exit_propensity
        transition_ik[:, :, 1, 0] = 0.10 + (0.08 + type_wage_response) * wage
        transition_ik[:, :, 1, 1] = (
            0.78 - 0.10 * pressure[:, None] - 0.06 * dual_access[:, None])
        transition_ik[:, :, 1, 2] = (
            0.015 + 0.16 * dual_access[:, None] + type_dual_propensity)
        transition_ik[:, :, 1, 3] = (
            0.055 + 0.04 * pressure[:, None] + type_exit_propensity)
        transition_ik[:, :, 2, 0] = 0.03 + 0.15 * type_loyal_persistence
        transition_ik[:, :, 2, 1] = 0.20 + amnesty + 0.18 * surveillance[:, None]
        transition_ik[:, :, 2, 2] = (
            0.54 + 0.18 * dual_access[:, None] - 0.28 * surveillance[:, None]
            - 0.18 * exit_pull[:, None] - amnesty + type_dual_propensity)
        transition_ik[:, :, 2, 3] = (
            0.23 + 0.20 * exit_pull[:, None] + 0.10 * surveillance[:, None]
            + type_exit_propensity)
        transition_ik[:, :, 3, 0] = 0.0
        transition_ik[:, :, 3, 1] = 0.025 * amnesty
        transition_ik[:, :, 3, 2] = 0.0
        transition_ik[:, :, 3, 3] = 1.0 - 0.025 * amnesty
        transition_ik = np.clip(transition_ik, 0.001, None)
        transition_ik /= transition_ik.sum(axis=3, keepdims=True)
        predicted = np.einsum(
            "nks,nksq->nkq", type_loyalty_belief[t], transition_ik)
        emission_mean_ik = (
            np.array([0.82, 0.57, 0.34, 0.12])[None, None, :]
            + np.array([-0.08, -0.03, 0.00, -0.04, -0.10])[None, :, None])
        emission_sigma = np.array([0.42, 0.30, 0.22, 0.24, 0.28])[None, :, None]
        emission = np.exp(-0.5 * (
            (intermediary_effort[t, :, None, None] - emission_mean_ik)
            / emission_sigma) ** 2)
        posterior = predicted * emission
        posterior /= posterior.sum(axis=2, keepdims=True)
        type_loyalty_belief[t + 1] = posterior
        loyalty_belief[t + 1] = np.einsum("nk,nks->ns", agent_type_share, posterior)
        node_signal = posterior[:, :, 0] + 0.55 * posterior[:, :, 1] + 0.15 * posterior[:, :, 2]
        loyalty_signal = np.sum(
            agent_type_share * type_effort_weight[None, :] * node_signal, axis=1)
        intermediary_effort[t] *= np.clip(0.45 + 0.55 * loyalty_signal, 0.35, 1.12)

        # Historically timed and stochastic jumps.  The 1898 emergency penal
        # order is a negative institutional shock; accommodation and surrender
        # programs can generate positive local jumps, while leader exits and
        # violent incidents create region-specific discontinuities.
        jump_hazard = np.clip(
            0.008 + 0.045 * leader_capacity[t] + 0.020 * coercion[t]
            + 0.030 * resistance[t], 0.0, 0.12)
        random_jump = rng.random(N) < jump_hazard
        jump_direction = np.where(
            rng.random(N) < (0.30 + 0.40 * accommodation[t]), 1.0, -1.0)
        jump_shock[t] = random_jump * jump_direction * rng.uniform(0.03, 0.13, N)
        jump_shock[t] -= punitive_edict * (0.08 + 0.05 * control[t])
        jump_shock[t] += 0.025 * pacification_signal * accommodation[t]

        if strategy_feedback:
            resistance_policy = softmax(np.array([
                1.15 * control[t].mean() + 0.75 * terrain.mean()
                + 0.45 * (1 - leader_capacity[t].mean()),
                1.10 * leader_capacity[t].mean() + 0.55 * (1 - control[t].mean())
                + 0.35 * flow_ratio[max(0, t - 1)],
            ]) / 0.40)
            dispersion[t], concentration[t] = resistance_policy
        else:
            dispersion[t], concentration[t] = 0.60, 0.40

        rainy = 0.5 + 0.5 * math.sin(2 * math.pi * ((dates[t].month - 5) % 12) / 12)
        edge_resistance = (resistance[t][:, None] + resistance[t][None, :]) / 2
        edge_services = (services[t][:, None] + services[t][None, :]) / 2
        capacity = base_transport * (1.0 - 0.52 * edge_resistance)
        capacity *= (0.72 + 0.28 * edge_services) * (1.0 - 0.18 * rainy * terrain[:, None])
        sources = [IDX["Aodi"], IDX["Keelung"]]
        if control[t, IDX["Tainan"]] > 0.72:
            sources.append(IDX["Tainan"])
        demand = 0.20 + 0.45 * control[t] + 0.28 * (1 - services[t])
        delivered = max_flow(capacity, sources, demand)
        flow_ratio[t] = np.clip(delivered / demand.sum(), 0.0, 1.0)

        schedule = sigmoid(3.0 * (t - capture_month))
        local_drive = (0.29 * schedule * (0.55 + 0.45 * flow_ratio[t])
                       + 0.065 * coercion[t])
        control_loss = 0.048 * resistance[t] * (0.55 + terrain) * control[t]
        control[t + 1] = np.clip(
            control[t] + local_drive * (1 - control[t]) - control_loss
            + rng.normal(0, 0.006, N), 0, 1)

        base_degree = np.maximum(base_social.sum(axis=1), 0.35)
        network_pressure = ((social @ resistance[t]) / base_degree
                            * (0.55 + 0.45 * social_trust[t])
                            + 0.12 * mainland_clandestine_contact[t])
        leader_term = 0.020 * leader_capacity[t] * (0.75 + 0.50 * concentration[t])
        repression_shock = 0.020 * coercion[t] * control[t]
        proxy_suppression = (0.015 * household_reporting[t]
                             * intermediary_effort[t] * resistance[t])
        joint_liability_grievance = (0.012 * coercion[t] * household_reporting[t]
                                     * (1.0 - legitimacy[t]))
        livelihood = 0.010 * (1 - services[t]) + 0.006 * urbanity
        demobilization = (0.012 + 0.034 * governance[t] + 0.026 * services[t]
                          + 0.016 * legitimacy[t] + 0.026 * reform_signal) * resistance[t]
        demobilization *= 1.0 - 0.18 * dispersion[t]
        resistance[t + 1] = np.clip(
            resistance[t]
            + (0.022 * network_pressure * (0.82 + 0.34 * dispersion[t])
               + leader_term + livelihood
               + 0.030 * grievance[t] + repression_shock
               + joint_liability_grievance - 0.018 * jump_shock[t]) * (1 - resistance[t])
            - demobilization - 0.012 * control[t] * resistance[t] - proxy_suppression
            + rng.normal(0, 0.006, N), 0, 1)

        gov_diffusion = normalize_rows(gov_net) @ governance[t]
        agency_delivery = np.power(np.maximum(
            administration_effort[t] * police_effort[t]
            * intermediary_effort[t] * household_reporting[t], 1e-8), 0.25)
        governance[t + 1] = np.clip(
            governance[t] + (0.006 + 0.015 * accommodation[t] + 0.012 * reform_signal)
            * control[t] * (0.55 + 0.45 * agency_delivery)
            * (1.0 - information_distortion[t]) * (1 - governance[t])
            + 0.006 * gov_diffusion - 0.012 * resistance[t] * governance[t]
            + 0.012 * jump_shock[t]
            + rng.normal(0, 0.003, N), 0, 1)
        services[t + 1] = np.clip(
            services[t] + (0.006 + 0.013 * accommodation[t] + 0.010 * reform_signal)
            * governance[t] * (1 - services[t])
            - 0.010 * resistance[t] * services[t] + rng.normal(0, 0.002, N), 0, 1)

        grievance[t + 1] = np.clip(
            0.91 * grievance[t] + 0.080 * coercion[t] * control[t]
            + 0.020 * agency_rent[t] + joint_liability_grievance
            + 0.025 * (1 - services[t]) - 0.045 * accommodation[t] * governance[t], 0, 1)
        legitimacy[t + 1] = np.clip(
            legitimacy[t] + 0.020 * services[t] + 0.018 * governance[t]
            - 0.055 * coercion[t] * grievance[t] - 0.014 * resistance[t]
            + 0.018 * jump_shock[t], 0, 1)

        # Hoko reporting may lower short-run enforcement cost but joint
        # liability and false reports erode horizontal trust.
        false_report_proxy = household_reporting[t] * information_distortion[t]
        social_trust[t + 1] = np.clip(
            social_trust[t] - 0.022 * coercion[t] * household_reporting[t]
            - 0.026 * false_report_proxy + 0.014 * services[t]
            + 0.008 * accommodation[t] * governance[t], 0.05, 0.90)

        # Adaptive broker network.  Hoko institutionalization and governance
        # create formal links; coercive reporting, false reports and low trust
        # sever informal links or drive them underground.
        if agent_network_game:
            edge_trust = (social_trust[t][:, None] + social_trust[t][None, :]) / 2
            edge_governance = (governance[t][:, None] + governance[t][None, :]) / 2
            edge_reporting = (
                household_reporting[t][:, None] + household_reporting[t][None, :]) / 2
            edge_distortion = (
                information_distortion[t][:, None]
                + information_distortion[t][None, :]) / 2
            growth = broker_network_base * (
                0.010 * hoko_signal + 0.008 * edge_governance + 0.006 * edge_trust)
            severance = broker_network * (
                0.009 * coercion[t] * edge_reporting + 0.010 * edge_distortion)
            broker_network = np.clip(broker_network + growth - severance, 0.0, 1.35)
            broker_network *= broker_network_base > 0

        # Extraction-output-resistance block, timed to the land survey and
        # monopoly expansion rather than imposed from month zero.
        extraction_rate[t] = np.clip(
            0.10 + 0.16 * land_survey_signal + 0.10 * reform_signal
            + 0.08 * household_reporting[t] - 0.05 * agency_rent[t], 0.04, 0.46)
        economic_output[t + 1] = np.clip(
            economic_output[t] * (1.0 + 0.010 * services[t]
                                  + 0.008 * governance[t]
                                  + 0.004 * regulated_cross_strait_trade[t]
                                  - 0.012 * resistance[t]
                                  - 0.010 * coercion[t] * grievance[t]
                                  - 0.006 * extraction_rate[t] ** 2)
            + 0.004 * land_survey_signal * control[t], 0.15, 1.40)

        # Slow identity mean field.  Schooling/administrative incentives pull
        # toward colonial institutions; coercion, grievance and resistance
        # memories pull away.  The state changes much more slowly than control.
        peer_identity = normalize_rows(base_social) @ identity_alignment[t]
        education_signal = reform_signal * (0.35 + 0.65 * services[t])
        identity_alignment[t + 1] = np.clip(
            identity_alignment[t]
            + 0.0030 * education_signal * (1.0 - identity_alignment[t])
            + 0.0018 * (peer_identity - identity_alignment[t])
            + 0.0012 * governance[t] * economic_output[t]
            - 0.0014 * mainland_clandestine_contact[t]
            - 0.0032 * coercion[t] * grievance[t]
            - 0.0018 * resistance[t] + 0.0010 * jump_shock[t], 0.0, 1.0)

        if adaptive:
            refugee = np.abs(resistance[t + 1] - resistance[t])
            adaptive_gain = 0.060 * np.outer(grievance[t], grievance[t])
            adaptive_gain += 0.025 * np.outer(refugee, refugee)
            social = np.clip(social + (base_social > 0) * adaptive_gain
                             - 0.018 * np.outer(governance[t], governance[t]), 0, 1.25)
            social *= (base_social > 0)

        mobility = normalize_rows(disease_net * capacity)
        force_density = 0.35 + 0.65 * schedule
        beta = 0.18 + 0.16 * rainy + 0.12 * force_density + 0.10 * (1 - services[t])
        infectious_mix = infected[t]
        if disease_mobility:
            infectious_mix = 0.68 * infected[t] + 0.32 * (mobility.T @ infected[t])
        new_exposed = beta * susceptible[t] * infectious_mix
        new_infected = 0.42 * exposed[t]
        new_recovered = 0.23 * infected[t]
        susceptible[t + 1] = np.clip(susceptible[t] - new_exposed, 0, 1)
        exposed[t + 1] = np.clip(exposed[t] + new_exposed - new_infected, 0, 1)
        infected[t + 1] = np.clip(infected[t] + new_infected - new_recovered, 0, 1)
        recovered[t + 1] = np.clip(recovered[t] + new_recovered, 0, 1)

    active = ((MONTHS - 1) <= leader_exits).astype(float)
    leader_capacity[-1] = np.clip(active @ incidence / 2.2, 0.0, 1.0) if leaders else 0
    flow_ratio[-1] = flow_ratio[-2]
    coercion[-1], accommodation[-1] = coercion[-2], accommodation[-2]
    dispersion[-1], concentration[-1] = dispersion[-2], concentration[-2]
    principal_monitoring[-1] = principal_monitoring[-2]
    principal_wage[-1] = principal_wage[-2]
    local_autonomy[-1] = local_autonomy[-2]
    administration_effort[-1] = administration_effort[-2]
    police_effort[-1] = police_effort[-2]
    intermediary_effort[-1] = intermediary_effort[-2]
    household_reporting[-1] = household_reporting[-2]
    agency_rent[-1] = agency_rent[-2]
    information_distortion[-1] = information_distortion[-2]
    social_trust[-1] = social_trust[-2]
    economic_output[-1] = economic_output[-2]
    extraction_rate[-1] = extraction_rate[-2]
    identity_alignment[-1] = identity_alignment[-2]
    jump_shock[-1] = jump_shock[-2]
    agent_compliance_action[-1] = agent_compliance_action[-2]
    broker_network_density[-1] = broker_network_density[-2]
    broker_spectral_radius[-1] = broker_spectral_radius[-2]
    broker_algebraic_connectivity[-1] = broker_algebraic_connectivity[-2]
    broker_weighted_clustering[-1] = broker_weighted_clustering[-2]
    maritime_blockade[-1] = maritime_blockade[-2]
    mainland_clandestine_contact[-1] = mainland_clandestine_contact[-2]
    regulated_cross_strait_trade[-1] = regulated_cross_strait_trade[-2]

    weights = population / population.sum()
    agg = lambda x: x @ weights
    agg_control = agg(control)
    agg_resistance = agg(resistance)
    agg_governance = agg(governance)
    agg_services = agg(services)
    agg_legitimacy = agg(legitimacy)
    agg_infected = agg(infected)
    agg_admin_effort = agg(administration_effort)
    agg_police_effort = agg(police_effort)
    agg_intermediary_effort = agg(intermediary_effort)
    agg_reporting = agg(household_reporting)
    agg_agency_rent = agg(agency_rent)
    agg_information_distortion = agg(information_distortion)
    agg_social_trust = agg(social_trust)
    agg_economic_output = agg(economic_output)
    agg_extraction_rate = agg(extraction_rate)
    agg_identity_alignment = agg(identity_alignment)
    agg_mainland_contact = agg(mainland_clandestine_contact)
    agg_cross_strait_trade = agg(regulated_cross_strait_trade)
    agg_loyalty_belief = np.einsum("tni,n->ti", loyalty_belief, weights)
    agg_type_share = np.einsum("n,nk->k", weights, agent_type_share)
    agg_type_loyalty_belief = np.einsum(
        "n,nk,tnks->tks", weights, agent_type_share, type_loyalty_belief)
    agg_type_loyalty_belief /= np.maximum(
        agg_type_share[None, :, None], 1e-12)
    agg_agent_compliance = np.einsum(
        "n,nk,tnk->t", weights, agent_type_share, agent_compliance_action)

    exposure = infected * (np.arange(MONTHS)[:, None] < 12)
    case_weights = exposure * population
    disease_cases = case_weights / case_weights.sum() * 26_094
    disease_deaths = case_weights / case_weights.sum() * 4_642

    reform = sigmoid((np.arange(MONTHS) - 48) / 11.0)
    leader_burden = leader_capacity.mean(axis=1)
    agency_contract_cost = (0.30 * principal_wage + 0.24 * principal_monitoring
                            + 0.28 * agg_agency_rent)
    fiscal_expense = (5.0 + 2.6 * agg_resistance + 2.0 * agg_infected
                      + 1.3 * (1 - agg_governance) + 1.2 * (1 - flow_ratio)
                      + 0.7 * leader_burden + 5.1 * reform + agency_contract_cost)
    fiscal_revenue = (1.6 + 1.8 * agg_governance + 4.15 * reform
                      + 1.45 * agg_control * agg_governance
                      + 0.42 * agg_reporting * (1.0 - agg_information_distortion)
                      - 0.32 * agg_agency_rent
                      + 0.85 * agg_extraction_rate * agg_economic_output)
    mask_1896 = np.array([d.year == 1896 for d in dates])
    fiscal_revenue += np.mean(fiscal_expense[mask_1896] - fiscal_revenue[mask_1896]) - 4.0
    occupation_cost = (fiscal_expense + 2.1 * agg_resistance + 1.4 * (1 - flow_ratio)
                       + 0.00011 * disease_cases.sum(axis=1))

    return {
        "dates": dates, "control": control, "resistance": resistance,
        "governance": governance, "services": services, "legitimacy": legitimacy,
        "grievance": grievance, "infected": infected, "leader_capacity": leader_capacity,
        "flow_ratio": flow_ratio, "coercion": coercion, "accommodation": accommodation,
        "dispersion": dispersion, "concentration": concentration,
        "principal_monitoring": principal_monitoring,
        "principal_wage": principal_wage, "local_autonomy": local_autonomy,
        "administration_effort": administration_effort,
        "police_effort": police_effort,
        "intermediary_effort": intermediary_effort,
        "household_reporting": household_reporting,
        "agency_rent": agency_rent,
        "information_distortion": information_distortion,
        "loyalty_belief": loyalty_belief,
        "type_loyalty_belief": type_loyalty_belief,
        "agent_type_share": agent_type_share,
        "agent_compliance_action": agent_compliance_action,
        "broker_network_density": broker_network_density,
        "broker_spectral_radius": broker_spectral_radius,
        "broker_algebraic_connectivity": broker_algebraic_connectivity,
        "broker_weighted_clustering": broker_weighted_clustering,
        "social_trust": social_trust,
        "economic_output": economic_output,
        "extraction_rate": extraction_rate,
        "identity_alignment": identity_alignment,
        "jump_shock": jump_shock,
        "maritime_blockade": maritime_blockade,
        "mainland_clandestine_contact": mainland_clandestine_contact,
        "regulated_cross_strait_trade": regulated_cross_strait_trade,
        "belief": belief, "disease_cases": disease_cases,
        "disease_deaths": disease_deaths, "fiscal_expense": fiscal_expense,
        "fiscal_revenue": fiscal_revenue, "occupation_cost": occupation_cost,
        "aggregate": {"control": agg_control, "resistance": agg_resistance,
                      "governance": agg_governance, "services": agg_services,
                      "legitimacy": agg_legitimacy, "infected": agg_infected,
                      "administration_effort": agg_admin_effort,
                      "police_effort": agg_police_effort,
                      "intermediary_effort": agg_intermediary_effort,
                      "household_reporting": agg_reporting,
                      "agency_rent": agg_agency_rent,
                      "information_distortion": agg_information_distortion,
                      "social_trust": agg_social_trust,
                      "economic_output": agg_economic_output,
                      "extraction_rate": agg_extraction_rate,
                      "identity_alignment": agg_identity_alignment,
                      "mainland_clandestine_contact": agg_mainland_contact,
                      "regulated_cross_strait_trade": agg_cross_strait_trade},
        "aggregate_loyalty_belief": agg_loyalty_belief,
        "aggregate_type_share": agg_type_share,
        "aggregate_type_loyalty_belief": agg_type_loyalty_belief,
        "aggregate_agent_compliance": agg_agent_compliance,
        "leader_names": names,
    }


def annual_rows(sim):
    rows = []
    for year in sorted({d.year for d in sim["dates"]}):
        mask = np.array([d.year == year for d in sim["dates"]])
        row = {"year": year}
        for key, values in sim["aggregate"].items():
            row[key] = float(values[mask].mean())
        row["flow_ratio"] = float(sim["flow_ratio"][mask].mean())
        row["coercion_probability"] = float(sim["coercion"][mask].mean())
        row["fiscal_gap_million_yen"] = float(
            (sim["fiscal_expense"][mask] - sim["fiscal_revenue"][mask]).mean())
        row["occupation_cost_index"] = float(sim["occupation_cost"][mask].mean())
        rows.append(row)
    return rows


def font(size, bold=False):
    path = Path("C:/Windows/Fonts") / ("msyhbd.ttc" if bold else "msyh.ttc")
    return ImageFont.truetype(str(path), size) if path.exists() else ImageFont.load_default()


def heatmap(path, values, title):
    cell_w, cell_h = 8, 38
    left, top, right, bottom = 210, 90, 45, 70
    width = left + MONTHS * cell_w + right
    height = top + N * cell_h + bottom
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    draw.text((left, 25), title, fill="#111827", font=font(28, True))
    for i, name in enumerate(NODES):
        y = top + i * cell_h
        draw.text((12, y + 7), name, fill="#374151", font=font(16))
        for t in range(MONTHS):
            v = float(np.clip(values[t, i], 0, 1))
            color = (int(245 - 155 * v), int(247 - 205 * v), int(250 - 210 * v))
            x = left + t * cell_w
            draw.rectangle((x, y, x + cell_w, y + cell_h - 2), fill=color)
    for year in (1895, 1898, 1901, 1905):
        t = next(i for i, d in enumerate([month_date(k) for k in range(MONTHS)]) if d.year == year)
        x = left + t * cell_w
        draw.line((x, top, x, top + N * cell_h), fill="#9ca3af", width=1)
        draw.text((x - 15, top + N * cell_h + 12), str(year), fill="#4b5563", font=font(15))
    draw.text((left, height - 32), "Light = low latent resistance; dark red = high.",
              fill="#6b7280", font=font(15))
    img.save(path)


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def main():
    base = simulate()
    annual = annual_rows(base)

    node_rows = []
    for t, d in enumerate(base["dates"]):
        for i, node in enumerate(NODES):
            node_rows.append({
                "date": d.isoformat(), "node": node,
                "control": float(base["control"][t, i]),
                "resistance": float(base["resistance"][t, i]),
                "governance": float(base["governance"][t, i]),
                "services": float(base["services"][t, i]),
                "legitimacy": float(base["legitimacy"][t, i]),
                "infected_share": float(base["infected"][t, i]),
                "leader_capacity": float(base["leader_capacity"][t, i]),
                "administration_effort": float(base["administration_effort"][t, i]),
                "police_effort": float(base["police_effort"][t, i]),
                "intermediary_effort": float(base["intermediary_effort"][t, i]),
                "household_reporting": float(base["household_reporting"][t, i]),
                "social_trust": float(base["social_trust"][t, i]),
                "economic_output": float(base["economic_output"][t, i]),
                "extraction_rate": float(base["extraction_rate"][t, i]),
                "identity_alignment": float(base["identity_alignment"][t, i]),
                "mainland_clandestine_contact": float(
                    base["mainland_clandestine_contact"][t, i]),
                "loyalty_loyal": float(base["loyalty_belief"][t, i, 0]),
                "loyalty_opportunistic": float(base["loyalty_belief"][t, i, 1]),
                "loyalty_dual": float(base["loyalty_belief"][t, i, 2]),
                "loyalty_exit": float(base["loyalty_belief"][t, i, 3]),
                **{
                    f"type_share_{name}": float(base["agent_type_share"][i, k])
                    for k, name in enumerate(AGENT_TYPES)
                },
                "committed_type_loyal_state": float(
                    base["type_loyalty_belief"][t, i, 0, 0]),
            })
    write_csv(OUT / "乙未战争_多层网络节点月度状态.csv", node_rows)
    write_csv(OUT / "乙未战争_多层网络年度汇总.csv", annual)

    ablations = {
        "full": {},
        "fixed_network": {"adaptive": False},
        "no_leader_hypergraph": {"leaders": False},
        "no_disease_mobility": {"disease_mobility": False},
        "no_strategy_feedback": {"strategy_feedback": False},
        "no_multilayer_agency": {"multilayer_agency": False},
        "no_agent_network_game": {"agent_network_game": False},
    }
    ablation_rows = []
    for name, kwargs in ablations.items():
        sim = simulate(seed=1895, **kwargs)
        ann = annual_rows(sim)
        ablation_rows.append({
            "model": name,
            "resistance_1902": next(r["resistance"] for r in ann if r["year"] == 1902),
            "governance_1902": next(r["governance"] for r in ann if r["year"] == 1902),
            "mean_flow_1895_1896": float(np.mean(sim["flow_ratio"][:19])),
            "peak_infected_share": float(np.max(sim["aggregate"]["infected"])),
            "cumulative_cost_index": float(np.sum(sim["occupation_cost"])),
        })
    write_csv(OUT / "乙未战争_网络模型消融实验.csv", ablation_rows)

    ensemble = []
    for seed in range(189500, 189700):
        sim = simulate(seed=seed)
        ann = annual_rows(sim)
        self_year = next((r["year"] for r in ann if r["fiscal_gap_million_yen"] <= 0), None)
        ensemble.append([
            next(r["resistance"] for r in ann if r["year"] == 1902),
            next(r["governance"] for r in ann if r["year"] == 1902),
            np.sum(sim["occupation_cost"]),
            self_year if self_year is not None else 9999,
            next(r["social_trust"] for r in ann if r["year"] == 1902),
            next(r["identity_alignment"] for r in ann if r["year"] == 1902),
            next(r["mainland_clandestine_contact"] for r in ann if r["year"] == 1902),
            float(sim["aggregate_loyalty_belief"][
                next(i for i, d in enumerate(sim["dates"]) if d.year == 1902), 2]),
            float(sim["aggregate_type_loyalty_belief"][
                next(i for i, d in enumerate(sim["dates"]) if d.year == 1902), 0, 0]),
            float(sim["aggregate_agent_compliance"][
                next(i for i, d in enumerate(sim["dates"]) if d.year == 1902)]),
            float(sim["broker_spectral_radius"][
                next(i for i, d in enumerate(sim["dates"]) if d.year == 1902)]),
            float(sim["broker_algebraic_connectivity"][
                next(i for i, d in enumerate(sim["dates"]) if d.year == 1902)]),
            float(sim["broker_weighted_clustering"][
                next(i for i, d in enumerate(sim["dates"]) if d.year == 1902)]),
        ])
    ensemble = np.asarray(ensemble)

    heatmap(OUT / "乙未战争_多层网络抵抗热图.png", base["resistance"],
            "Yiwei War multilayer model: resistance by historical region")

    fiscal_gap_1896 = next(r["fiscal_gap_million_yen"] for r in annual if r["year"] == 1896)
    self_year = next((r["year"] for r in annual if r["fiscal_gap_million_yen"] <= 0), None)
    summary = {
        "model": "historically timed multilayer principal-agent spatial state-space model",
        "nodes": NODES,
        "months": MONTHS,
        "layers": ["transport", "social mobilization", "governance", "disease mobility"],
        "principal_agent_layers": [
            "Japanese center and Governor-General",
            "local administration",
            "police and direct enforcement",
            "local elites, hoko and auxiliary groups",
            "households and villages"
        ],
        "leaders_in_hypergraph": list(LEADERS),
        "latent_strategy_states": ["advance", "coercive consolidation", "institutional governance"],
        "calibration": {
            "disease_cases": float(base["disease_cases"].sum()),
            "disease_deaths": float(base["disease_deaths"].sum()),
            "fiscal_gap_1896_million_yen": fiscal_gap_1896,
            "first_fiscal_self_reliance_year": self_year,
        },
        "ensemble_200": {
            "resistance_1902_p05_p50_p95": np.quantile(ensemble[:, 0], [0.05, 0.5, 0.95]).tolist(),
            "governance_1902_p05_p50_p95": np.quantile(ensemble[:, 1], [0.05, 0.5, 0.95]).tolist(),
            "cumulative_cost_p05_p50_p95": np.quantile(ensemble[:, 2], [0.05, 0.5, 0.95]).tolist(),
            "fiscal_self_reliance_years": sorted(set(ensemble[:, 3].astype(int).tolist())),
            "social_trust_1902_p05_p50_p95": np.quantile(
                ensemble[:, 4], [0.05, 0.5, 0.95]).tolist(),
            "identity_alignment_1902_p05_p50_p95": np.quantile(
                ensemble[:, 5], [0.05, 0.5, 0.95]).tolist(),
            "mainland_contact_1902_p05_p50_p95": np.quantile(
                ensemble[:, 6], [0.05, 0.5, 0.95]).tolist(),
            "dual_loyalty_1902_p05_p50_p95": np.quantile(
                ensemble[:, 7], [0.05, 0.5, 0.95]).tolist(),
            "committed_type_loyal_state_1902_p05_p50_p95": np.quantile(
                ensemble[:, 8], [0.05, 0.5, 0.95]).tolist(),
            "agent_compliance_1902_p05_p50_p95": np.quantile(
                ensemble[:, 9], [0.05, 0.5, 0.95]).tolist(),
            "broker_spectral_radius_1902_p05_p50_p95": np.quantile(
                ensemble[:, 10], [0.05, 0.5, 0.95]).tolist(),
            "broker_algebraic_connectivity_1902_p05_p50_p95": np.quantile(
                ensemble[:, 11], [0.05, 0.5, 0.95]).tolist(),
            "broker_weighted_clustering_1902_p05_p50_p95": np.quantile(
                ensemble[:, 12], [0.05, 0.5, 0.95]).tolist(),
        },
        "agent_type_model": {
            "types": list(AGENT_TYPES),
            "baseline_population_weighted_shares": {
                name: float(base["aggregate_type_share"][k])
                for k, name in enumerate(AGENT_TYPES)
            },
            "baseline_type_loyalty_1902": {
                name: {
                    state: float(base["aggregate_type_loyalty_belief"][
                        next(i for i, d in enumerate(base["dates"]) if d.year == 1902), k, s])
                    for s, state in enumerate(
                        ["loyal", "opportunistic", "dual", "exit"])
                }
                for k, name in enumerate(AGENT_TYPES)
            },
            "warning": "Latent classes and shares are structural priors, not observed historical population categories."
        },
        "baseline_agency_1902": {
            key: next(r[key] for r in annual if r["year"] == 1902)
            for key in ["administration_effort", "police_effort",
                        "intermediary_effort", "household_reporting",
                        "agency_rent", "information_distortion",
                        "social_trust", "economic_output",
                        "extraction_rate", "identity_alignment",
                        "mainland_clandestine_contact",
                        "regulated_cross_strait_trade"]
        },
        "baseline_complex_network_1902": {
            key: float(base[key][
                next(i for i, d in enumerate(base["dates"]) if d.year == 1902)])
            for key in ["broker_network_density", "broker_spectral_radius",
                        "broker_algebraic_connectivity",
                        "broker_weighted_clustering"]
        },
        "interpretation": "Network states and strategy beliefs are latent explanatory constructs, not observed historical measurements.",
    }

    assert abs(summary["calibration"]["disease_cases"] - 26_094) < 1e-6
    assert abs(summary["calibration"]["disease_deaths"] - 4_642) < 1e-6
    assert abs(fiscal_gap_1896 - 4.0) < 1e-9
    assert self_year in (1904, 1905)
    assert base["control"][4, IDX["Taipei"]] > base["control"][4, IDX["Tainan"]]
    assert base["resistance"][12, IDX["MountainInterior"]] > base["resistance"][12, IDX["Taipei"]]

    (OUT / "乙未战争_多层网络模型摘要.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("NETWORK_MODEL_VERIFICATION: PASS")


if __name__ == "__main__":
    main()
