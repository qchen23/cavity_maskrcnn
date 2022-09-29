from multiprocessing import Pool
import argparse
import os
import math
import subprocess


# the function to call QC_bubble/bubble_detection.py
def run_detection(params):
  (proc_id, i, num_detect,  model, data_dir, output_dir) = params
  command = "python -u QC_bubble/bubble_detection.py -o {} -d {} -m {} -s {} -n {}".format(output_dir, data_dir, model, i, num_detect)
  # out = subprocess.check_output(command, shell=True).decode("utf-8")
  # print(out, flush = True)
  proc = subprocess.Popen(command.split(" "),stdout=subprocess.PIPE)
  while True:
    line = proc.stdout.readline()
    if not line: break
    print("Processor {} : {}".format(proc_id, line.rstrip()))

  # subprocess.Popen(command, shell=True)


if __name__ == '__main__':    
  parser = argparse.ArgumentParser(description="Bubble detection")
  parser.add_argument("--output_dir", "-o", required=True, type=str, help="output directory to store the checkpoint file")
  parser.add_argument("--dataset", "-d", required = True, type=str, help="data_set to do the detection")
  parser.add_argument("--model", "-m", required = True, type=str, help="model to run")
  parser.add_argument("--processor", "-p", required = True, type=int, help="number of processors")
  args = parser.parse_args()

  params = []
  num_processors = args.processor
  
  data_dir = args.dataset
  filenames = os.listdir(data_dir)

  num_detect = math.ceil(len(filenames) / num_processors)
  for i in range(0, len(filenames), num_detect):
    params.append((int(i/num_detect), i, num_detect,  args.model, data_dir, args.output_dir))
  
  print(len(params))
  pool = Pool()
  pool.map(run_detection, params)

