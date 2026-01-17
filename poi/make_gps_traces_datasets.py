import argparse

from datasets import Dataset
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from geopy.distance import geodesic


def main(args):
    dataset = []
    with open("traces.gpx", "r") as f:
        lines = f.readlines()

        for line in lines:
            if "<trk>" in line:
                curr_sample = {"name": None, "desc": None, "url": None, "path": []}
            elif "</trk>" in line:
                dataset.append(curr_sample)
            elif "<name>" in line:
                curr_sample["name"] = line.split("<name>")[-1].split("</name>")[0]
            elif "<desc>" in line:
                curr_sample["desc"] = line.split("<desc>")[-1].split("</desc>")[0]
            elif "<url>" in line:
                curr_sample["url"] = line.split("<url>")[-1].split("</url>")[0]
            elif "<trkpt " in line:
                curr_sample["path"].append(
                    (
                        float(line.split('lat="')[-1].split('" lon="')[0]),
                        float(line.split('" lon="')[-1].split('">')[0]),
                    )
                )

    hf_dataset = Dataset.from_list(dataset)
    hf_dataset.save_to_disk("gps_traces")

    print(hf_dataset[0])

    samples_length = []
    segments_length = []
    n_segments = []
    filtered_dataset = []
    for sample in hf_dataset:
        segment_length = []
        for tracking_point_a, tracking_point_b in zip(sample["path"], sample["path"][1:]):
            segment_length.append(geodesic((tracking_point_a[0], tracking_point_a[1]), (tracking_point_b[0], tracking_point_b[1])).meters)
        sample_length = sum(segment_length)
        if sample_length < 15000 and sample_length > 300 and len(segment_length) > 15 and max(segment_length) < 100:
            segments_length += segment_length
            samples_length.append(sample_length)
            n_segments.append(len(segment_length))
            filtered_dataset.append(sample)

    filtered_dataset = Dataset.from_list(filtered_dataset)

    plt.plot([point[0] for point in filtered_dataset[-1]["path"]], [point[1] for point in filtered_dataset[-1]["path"]], c="blue", alpha=0.5, linewidth=0.3)
    plt.show()
    plt.savefig("traces.png")

    fig, ax = plt.subplots()

    def animate(i):
        ax.clear()
        ax.plot([point[0] for point in filtered_dataset[i]["path"]], [point[1] for point in filtered_dataset[i]["path"]], c="blue", alpha=0.5, linewidth=0.3)
        ax.set_xlim(48.66, 48.71)
        ax.set_ylim(6.13, 6.22)

    ani = animation.FuncAnimation(fig, animate, frames=len(filtered_dataset), interval=100)
    ani.save("traces.gif", writer="pillow")
    plt.close()

    print("n kept", len(samples_length))
    plt.hist(samples_length, bins=30)
    plt.show()
    plt.savefig("samples_length.png")
    plt.hist(segments_length, bins=500)
    plt.show()
    plt.savefig("segments_length.png")
    plt.hist(n_segments, bins=30)
    plt.show()
    plt.savefig("n_segments.png")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Make gps traces datasets for training/testing.")
    parser.add_argument("--n_train_sample", type=int, default=10000, help="Number of samples to generate for train.")
    parser.add_argument("--n_test_sample", type=int, default=250, help="Number of samples to generate for test.")
    args = parser.parse_args()
    main(args) 