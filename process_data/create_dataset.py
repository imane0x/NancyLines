import argparse
import json
import random

from datasets import load_dataset


def get_line_det():
    dets = [
        "la ligne '{line}'",
        "le bus '{line}'",
        "le trajet '{line}'",
        "le trajet de la ligne '{line}'",
        "le trajet du bus '{line}'",
        "la ligne de bus '{line}'",
    ]
    return random.choice(dets)


def get_stop_det(rel):
    dets = [
        f"{{origin}} {rel} {{destination}}",
        f"l'arrêt {{origin}} {rel} l'arrêt {{destination}}",
        f"la station {{origin}} {rel} la station {{destination}}",
    ]
    if rel == "et":
        dets += [
            f"les arrêts {{origin}} {rel} {{destination}}",
            f"les stations {{origin}} {rel} {{destination}}",
        ]
    return random.choice(dets)


def get_ways_det():
    dets = [
        "{ways}",
    ]
    return random.choice(dets)


def get_template(order, is_plural):
    if order == ["line", "stops", "ways"]:
        line_to_stops = [
            "circule entre",
            "dessert",
            "relie",
            "connecte",
            "assure la liaison entre",
        ]
        stops_to_ways = [
            "en passant par",
            "via",
            "en traversant",
            "en suivant",
            "par",
        ]
        return f"{get_line_det()} {random.choice(line_to_stops)} {get_stop_det('et')} {random.choice(stops_to_ways)} {get_ways_det()}"

    elif order == ["line", "ways", "stops"]:
        line_to_ways = [
            "emprunte",
            "parcours",
            "traverse",
            "suit",
            "passe par",
        ]
        ways_to_stops = [
            "de",
            "dans la section reliant",
            "dans l'intervalle reliant",
            "pour relier",
            "pour connecter",
        ]
        return f"{get_line_det()} {random.choice(line_to_ways)} {get_ways_det()} {random.choice(ways_to_stops)} {get_stop_det('à')}"
        
    elif order == ["stops", "line", "ways"]:
        prefix = [
            f"entre {get_stop_det('et')}",
            f"pour aller de {get_stop_det('à')}",
            f"de {get_stop_det('à')}",
        ]
        line_to_ways = [
            "effectue son trajet en passant par",
            "emprunte",
            "assure la liaison en passant par",
            "circule via",
            "roule sur",
        ]
        return f"{random.choice(prefix)}, {get_line_det()} {random.choice(line_to_ways)} {get_ways_det()}"

    elif order == ["stops", "ways", "line"]:
        stops_to_ways = [
            "on trouve",
            "le trajet passe par",
            "l'itinéraire comprend",
            "le parcours traverse",
            "on traverse",
        ]
        ways_to_line = [
            "desservi par",
            "opéré par",
            "selon",
            "selon le tracé défini par",
            "selon le plan de",
        ]
        return f"entre {get_stop_det('et')} {random.choice(stops_to_ways)} {get_ways_det()} {random.choice(ways_to_line)} {get_line_det()}"

    elif order == ["ways", "line", "stops"]:
        if is_plural:
            ways_to_line = [
                "sont traversées par",
                "sont utilisées par",
                "font parti du tracé de",
                "desservent",
                "sont incluses dans l'itinéraire de",
            ]
        else:
            ways_to_line = [
                "est traversée par",
                "est utilsée par",
                "fait parti du tracé de",
                "dessert",
                "est incluse dans l'itinéraire de",
            ]
        line_to_stops = [
            "pour relier",
            "afin de connecter",
            "entre",
            "dans la section reliant",
        ]
        return f"{get_ways_det()} {random.choice(ways_to_line)} {get_line_det()} {random.choice(line_to_stops)} {get_stop_det('et')}"

    elif order == ["ways", "stops", "line"]:
        if is_plural:
            ways_to_stops = [
                "se situent entre",
                "relient",
                "permmettent la connexion entre",
                "se trouvent entre",
                "séparent",
            ]
        else:
            ways_to_stops = [
                "se situe entre",
                "relie",
                "permmet la connexion entre",
                "se trouve entre",
                "sépare"
            ]
        stops_to_line = [
            "de",
            "appartenant à",
            "qui font parti de",
            "dans le tracé de",
            "sur",
        ]
        return f"{get_ways_det()} {random.choice(ways_to_stops)} {get_stop_det('et')} {random.choice(stops_to_line)} {get_line_det()}"


def get_ways(ways):
    output = ""

    if len(ways) == 0:
        output = "aucune rue"

    for i, way in enumerate(ways):

        if way.lower().startswith(("allée", "avenue", "esplanade", "impasse")):
            det = "l'"
        elif way.lower().startswith(("route", "rue", "grande", "place", "street")):
            det = "la "
        else:
            det = "le "

        if i == len(ways) - 1:
            sep = ""
        elif i == len(ways) - 2:
            if len(ways) > 3:
                sep = random.choice([" et ", " et enfin ", " ainsi que "])
            else:
                sep = random.choice([" et "])
        else:
            sep = random.choice([", ", ", puis "])

        output += det + way + sep
        
    return output


def to_sentence(sample):
    order = ["ways", "stops", "line"]
    random.shuffle(order)
    is_plural = len(sample["ways"]) > 1
    template = get_template(order, is_plural)
    ways = get_ways(sample["ways"]) 
    sentence = template.format(ways=ways, origin=sample["from"], destination=sample["to"], line=sample["line"])
    sentence = sentence.replace(" de le ", " du ").replace(" à le ", "au")
    sentence = sentence[0].upper() + sentence[1:]
    return sentence


def main(args):
    data = load_dataset(args.input_data)["train"]

    # Uncomment to see type of ways in the dataset
    # print(set([way.split(" ")[0] for sample in data for way in sample["ways"] if not way.startswith("street_")]))

    sentences = []
    for _ in range(args.n_epochs):
        for sample in data:
            sentences.append(to_sentence(sample))

    with open(args.output_data, "w") as f:
        json.dump(sentences, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Get training data sentences from pairs')
    parser.add_argument('--input_data', type=str, default="im21/pairs", help='Path to input dataset')
    parser.add_argument('--output_data', type=str, default="./data/train.json", help='Path to created dataset')
    parser.add_argument('--n_epochs', type=str, default=10, help='Number of augmentation per input sample')
    args = parser.parse_args()
    main(args)