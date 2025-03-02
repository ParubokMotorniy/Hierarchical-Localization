import sys
import numpy as np

results_file = sys.argv[1]
limits = sys.argv[2].split(',')
repetitions = int(sys.argv[3])
solver_name = sys.argv[4]

predictions = {}

with open(results_file, "r") as f:
    entire_file = f.read().rstrip().split("\n")
    idx = 0

    while idx < len(entire_file):
        for iterations_upper_bound in limits:
            repetition_data = {}
            for i in range(repetitions):
                data = entire_file[idx].split()
                name = data[0].split('/')[-1]

                if name not in predictions.keys():
                    if not (iterations_upper_bound == limits[0] and i == 0):
                        raise RuntimeError("Stride mismatch!")
                    else:
                        predictions[name] = {bound: None for bound in limits}

                q, t, time = np.split(np.array(data[1:], float), [4, 7])
                t = t[:3]
                time = time[0]
                if_no_error = not np.all([i == 2 for i in q]) 

                if if_no_error:
                    repetition_data[time] = (q, t)

                idx += 1

            if len(repetition_data) == 0:
                predictions[name][iterations_upper_bound] = ([[1.0,0.0,0.0,0.0], [0.0,0.0,0.0]])
            else:
                median_time_key = sorted(repetition_data.keys())[int(np.floor(len(repetition_data) / 2))]
                predictions[name][iterations_upper_bound] = repetition_data[median_time_key]

for limit in limits:
    with open(f"{solver_name}_limit_{limit}.txt", "w") as output:
        for img_name in predictions.keys():
            q,t = predictions[img_name][limit]
            q = " ".join([str(elem) for elem in q])
            t = " ".join([str(elem) for elem in t])

            output.write(f"{img_name} {q} {t}\n")
