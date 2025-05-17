import os
import re
import json
import matplotlib.pyplot as plt
import numpy as np

# Function to extract metrics from a text file
def extract_metrics(filename):
    metrics = {}
    with open(filename, "r") as file:
        for line in file:
            match = re.search(r"(\\w+)\\s*[:=]\\s*([\\d.]+)", line)
            if match:
                key, value = match.groups()
                metrics[key] = float(value)
    return metrics

# Function to generate a report
def generate_report(metrics_collection):
    report = {}
    for metrics in metrics_collection:
        for key, value in metrics.items():
            if key not in report:
                report[key] = []
            report[key].append(value)

    # Calculate mean values
    summary = {key: np.mean(values) for key, values in report.items()}

    # Save to JSON
    with open("metrics_data.json", "w") as json_file:
        json.dump(report, json_file, indent=4)

    # Generate bar charts
    for key, values in report.items():
        plt.figure()
        plt.bar(range(len(values)), values, tick_label=[f"Set {i+1}" for i in range(len(values))])
        plt.xlabel("Datasets")
        plt.ylabel(key)
        plt.title(f"Bar Chart of {key}")
        plt.savefig(f"{key}_bar_chart.png")
        plt.close()

    # Generate summary report
    with open("summary_report.txt", "w") as report_file:
        report_file.write("Metrics Summary Report\\n")
        report_file.write("===============================\\n")
        for key, mean_value in summary.items():
            report_file.write(f"{key}: {mean_value:.2f}\\n")

def main():
    metrics_collection = []
    
    for dirpath, _, filenames in os.walk("."):
        for filename in filenames:
            if filename.endswith(".txt"):
                filepath = os.path.join(dirpath, filename)
                metrics = extract_metrics(filepath)
                metrics_collection.append(metrics)

    generate_report(metrics_collection)

if __name__ == "__main__":
    main()
