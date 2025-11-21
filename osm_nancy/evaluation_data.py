import random
from datasets import Dataset

import random
from datasets import Dataset

def generate_two_pair_questions(dataset, n_samples=20, seed=None, step=1):
    """
    Randomly generate questions combining a sample with a following sample
    within a flexible step (i+1, i+2, or i+3).

    Each question uses:
      - 'from' from the first sample
      - 'to' from a sample i+step ahead
      - merged 'ways' from both samples

    Args:
        dataset (datasets.Dataset): HF dataset split (e.g. dataset["train"]).
        n_samples (int): Number of combined examples to generate.
        seed (int, optional): Random seed for reproducibility.
        max_step (int): Maximum step forward to pick the second sample (default 3).

    Returns:
        list of dict: [{'question': ..., 'answer': ...}]
    """
    if seed is not None:
        random.seed(seed)

    templates = [
        "Entre {stop_1} et {stop_2}, par où passe la ligne {line} ?",
        "Quel trajet suit la ligne {line} entre {stop_1} et {stop_2} ?",
        "Quels arrêts ou rues la ligne {line} traverse entre {stop_1} et {stop_2} ?",
        "Quel est l’itinéraire de la ligne {line} reliant {stop_1} et {stop_2} ?",
        "Par quelles rues ou arrêts passe la ligne {line} de {stop_1} à {stop_2} ?",
    ]

    qa_pairs = []
    max_index = len(dataset) - step  # avoid out-of-range

    for _ in range(n_samples):
        i = random.randint(0, max_index)
        #step = random.randint(1, max_step)
        ex1 = dataset[i]
        ex2 = dataset[i + step]

        stop_1 = ex1["from"]
        stop_2 = ex2["to"]
        line = ex1["line"]

        merged_ways = []
        for w in [ex1["ways"], ex2["ways"]]:
            merged_ways.extend(w if isinstance(w, list) else [w])

        # remove duplicates but preserve order
        seen = set()
        merged_ways = [w for w in merged_ways if not (w in seen or seen.add(w))]

        template = random.choice(templates)
        question = template.format(line=line, stop_1=stop_1, stop_2=stop_2)
        answer = ", ".join(merged_ways)

        qa_pairs.append({"question": question, "answer": answer})

    return qa_pairs
def generate_intermediate_stop_questions(dataset, n_samples=20, seed=None):
    """
    Generate questions asking for intermediate stops given a bus line and two non-consecutive stops.

    Args:
        dataset (list of dicts or datasets.Dataset): entries with 'line', 'from', 'to'
        n_samples (int): number of QA pairs
        seed (int, optional): random seed

    Returns:
        list of dicts: [{'question': ..., 'answer': ...}]
    """
    if seed is not None:
        random.seed(seed)

    templates = [
        "Quels sont les arrêts intermédiaires de la ligne {line} entre {stop_1} et {stop_2} ?",
        "Donne-moi les arrêts que la ligne {line} traverse entre {stop_1} et {stop_2}",
        "Liste les arrêts situés entre {stop_1} et {stop_2} pour la ligne {line}",
    ]

    # Build route for each line
    line_to_route = {}
    for entry in dataset:
        line = entry["line"]
        if line not in line_to_route:
            line_to_route[line] = []
        # store the segment
        line_to_route[line].append((entry["from"], entry["to"]))

    # Convert segments into ordered route
    for line, segments in line_to_route.items():
        route = [segments[0][0]]
        visited = set(route)
        while len(route) < len(segments) + 1:
            for fr, to in segments:
                if fr == route[-1] and to not in visited:
                    route.append(to)
                    visited.add(to)
                    break
                elif to == route[-1] and fr not in visited:
                    route.append(fr)
                    visited.add(fr)
                    break
            else:
                break
        line_to_route[line] = route

    qa_pairs = []
    lines = list(line_to_route.keys())
    max_index = len(lines) - 1

    for _ in range(n_samples):
        # pick random line and template
        line = lines[random.randint(0, max_index)]
        route = line_to_route[line]
        if len(route) < 3:
            continue  # no intermediate stops possible

        # pick two stops non-consecutive
        idx_start = random.randint(0, len(route)-3)
        idx_end = random.randint(idx_start+2, len(route)-1)
        stop_1, stop_2 = route[idx_start], route[idx_end]
        intermediates = route[idx_start+1:idx_end]

        template = templates[random.randint(0, len(templates)-1)]
        question = template.format(line=line, stop_1=stop_1, stop_2=stop_2)
        answer = ", ".join(intermediates)

        qa_pairs.append({"question": question, "answer": answer})

    return qa_pairs


def generate_lines_from_stops_segments(dataset, n_samples=20, seed=None):
    """
    Generate questions asking which bus lines stop at given bus stops (from-to segments).

    Args:
        dataset (list of dicts): each entry has 'line', 'from', 'to'
        n_samples (int): number of QA pairs to generate
        seed (int, optional): random seed

    Returns:
        list of dicts: [{'question': ..., 'answer': ...}]
    """
    if seed is not None:
        random.seed(seed)

    templates = [
        "Quelles lignes de bus s'arrêtent à {stops} ?",
        "Quels bus passent par les arrêts {stops} ?",
        "Donne-moi les lignes qui desservent {stops}.",
    ]

    # Build mapping: stop -> set of lines
    stop_to_lines = {}
    for entry in dataset:
        line = entry["line"]
        for stop in [entry.get("from"), entry.get("to")]:
            if stop is not None:
                stop_to_lines.setdefault(stop, set()).add(line)

    stops = list(stop_to_lines.keys())
    if not stops:
        return []

    qa_pairs = []
    while len(qa_pairs) < n_samples:
        # Pick a random number of stops (1 to 3 or len(stops))
        num_stops = random.randint(1, min(3, len(stops)))
        selected_stops = random.sample(stops, num_stops)

        # Find lines that stop at all selected stops
        lines_sets = [stop_to_lines[stop] for stop in selected_stops]
        common_lines = set.intersection(*lines_sets)

        if not common_lines:
            continue  # try again

        lines_str = ", ".join(sorted(common_lines))
        stops_str = ", ".join(selected_stops)
        template = random.choice(templates)
        question = template.format(stops=stops_str)

        qa_pairs.append({"question": question, "answer": lines_str})

    return qa_pairs
  def generate_stops_from_line(dataset, n_samples=40, seed=None):
    """
    Generate questions asking for all bus stops for a given line.

    Args:
        dataset (list of dicts): each entry has 'line', 'from', 'to'
        n_samples (int): number of QA pairs to generate
        seed (int, optional): random seed

    Returns:
        list of dicts: [{'question': ..., 'answer': ...}]
    """
    if seed is not None:
        random.seed(seed)

    templates = [
        "Quels sont tous les arrêts de bus de la ligne {line} ?",
        "Donne-moi la liste complète des arrêts pour la ligne {line}",
        "Par quels arrêts passe la ligne {line} ?",
    ]

    # Build mapping: line -> set of stops
    line_to_stops = {}
    for entry in dataset:
        line = entry["line"]
        for stop in [entry.get("from"), entry.get("to")]:
            if stop is not None:
                line_to_stops.setdefault(line, set()).add(stop)

    lines = list(line_to_stops.keys())
    if not lines:
        return []

    qa_pairs = []
    while len(qa_pairs) < n_samples:
        line = random.choice(lines)
        stops = sorted(line_to_stops[line])
        stops_str = ", ".join(stops)
        template = random.choice(templates)
        question = template.format(line=line)
        qa_pairs.append({"question": question, "answer": stops_str})

    return qa_pairs
