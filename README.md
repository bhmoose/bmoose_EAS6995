
# Guide to using NCAR HPC systems

Most information from class discussions, assignment instructions, and [NCAR HPC Documentation (n.d.). University Corporation for Atmospheric Research. Retrieved March 27, 2025, from https://ncar-hpc-docs.readthedocs.io/en/latest/ ]


### Logging in to Derecho and Casper:

1) Open a terminal window
2) Use SSH to connect to Derecho (`ssh bmoose@derecho.hpc.ucar.edu`) or Casper (`ssh bmoose@casper.hpc.ucar.edu`)
3) Enter NCAR password and 2 factor authentication
4) Once logged into the systems, you are on the Derecho or Casper login nodes, not directly connected to either of the HPC systems.

### Creating a job script

1) Open a new script file by using `vim filename.pbs`
2) Edit the file by typing "i" to begin editing in vim, [esc] to end editing, ":wq" to quit and save, ":q" to quit without saving
3) Add a header for the file (e.g. `#!/bin/tcsh`)
4) Set up the PBS variables:
   * `#PBS -N jobname` (sets the name of the job to be scheduled/run to "jobname")
   * `#PBS -q queue_name` (sets the queue for the scheduled job to queue_name)
       * For casper, use the `casper` queue
       * For derecho, use the `main` queue for most jobs (this charges for usage of a full node even if only a part of it is requested)
           * Generally use `#PBS -l job_priority=regular` (1x charge) or  `#PBS -l job_priority=economy` (0.7x charge)
       * For jobs on derecho using few resources, could use `develop` queue (only charges for what is used, but has resource limitations)
   * `#PBS -A project_code` (sets the project code to project_code, for this class this is UCOR0090). This charges the core hours to the correct project.
   * `#PBS -l select=[num_nodes]:ncpus=[num_cpus]:mpiprocs=[num_mpiprocs]:ngpus=[num_gpus]:mem=[reqd_mem]` (sets requested resources)
       * Requests num_nodes nodes
       * For each node, requests num_cpus CPU cores, num_gpus GPUs
       * Still need to learn more about what MPI processes are and the meaning of requesting them
       * Requests reqd_mem RAM
   * `#PBS -l walltime=hh:mm:ss` (requests hh hours, mm minutes, ss seconds of wallclock time for the job. If time exceeded, will stop job execution)
   * `#PBS -j oe` (combines output and error files, if not included the run will produce a separate file for output and errors)
5) If submitting a job that is a python file, add the following lines:
   * `module load conda` (loads conda module so that environments can be loaded)
   * `conda activate env_name` (activates a conda environment called env_name)
   * `python my_file.py` (runs my_file.py in Python, this is the file that contains the code to be run on the HPC system, such as the DL model training)

### Submitting and tracking jobs

1) Submit the job script using `qsub filename.pbs` where filename.pbs is the name of the job script. This submits the job to the scheduler, which will originally put the job in the queue, then the job will run once the resources to run it are available.
2) To track the status of the job, use `qstat -u $USER` to see the currently running or queued jobs, their names and ID numbers, wall clock times, status (running or queued), requested resources, and duration.
3) To look at recently completed jobs, use `qhist -u $USER` to see recently completed jobs (names, ID numbers, run times, resources allocated)
4) To cancel a job that is currently queued or running, use `qdel jobid` where jobid is the job ID number associated with the job. Sometimes there is a bit of a delay between when this request is sent and when the job disappears from the qstat list.

### JupyterHub

JupyterHub allows you to connect to a Casper login node and run code (usually analysis or plotting code that does not require many resources). Files on the Glade filesystem can be accessed within code written on JupyterHub (since JupyterHub is remote - it is not working locally). Local files need to be copied over with `scp` to the Glade filesystem before they can be used in code run on JupyterHub. Upon logging in, there are options to work on a Casper login node (choose this for all non-resource-intensive tasks) or directly on Derecho and Casper (with options to configure resource allocations such as number of nodes, CPU cores, and GPUs). Working directly on Derecho or Casper through JupyterHub should not often be used, since most tasks for which JupyterHub is useful (running Jupyter notebooks to analyze results) do not require the supercomputers' resources and forgetting to end the job or set a wallclock limit could result in substantial core hour charges. 

### Checking Core Hour Usage

To check core hours used, log in to the UCAR Systems Accounting Manager (SAM) at [sam.ucar.edu](sam.ucar.edu). There is a lag of about 1 day, so core hours charged to a project do not appear on SAM until a day or so after they have been used.



   
   
    
  

