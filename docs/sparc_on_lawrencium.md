# Running SpaRC on Lawrencium

SpaRC is an [Apache Spark-based scalable genomic sequence clustering application](#references). SpaRC has been running successfully on [AWS EMR](https://aws.amazon.com/emr/), as well as on the [Bridges](https://www.psc.edu/bridges) supercomputer at PSC. In this tutorial, I describe how to run SpaRC on [Lawrencium](https://sites.google.com/a/lbl.gov/high-performance-computing-services-group/lbnl-supercluster/lawrencium).

## Spark On Demand on Lawrencium

Users can run Spark jobs on Lawrencium in [Spark On Demand (SOD)](https://sites.google.com/a/lbl.gov/high-performance-computing-services-group/getting-started/faq) fashion, in which a standalone Spark cluster will be created *on demand* in a Slurm job. Note that the Spark cluster will be running in [standalone mode](https://spark.apache.org/docs/latest/spark-standalone.html), so there will be no YARN cluster manager, nor HDFS. In lieu of HDFS, we'll use Lustre scratch for storage.

As of this writing, there is only Spark **2.1.0** available on Lawrencium. We may install a more up-to-date version in the near future.

## Building SpaRC

You'll build SpaRC against Spark 2.1.0. The source code of SpaRC is hosted on Bitbucket at <https://bitbucket.org/LizhenShi/sparc>. I don't have write access to the repo, so I imported it to GitHub, at <https://github.com/shawfdong/sparc>. I've added a new file `build.sbt.spark2.1.0` to the GitHub repo, which, as the name suggests, will be used to build SpaRC against Spark 2.1.0.

Note that you can't build SpaRC on the login nodes of Lawrencium, because rsync (which is required by sbt) is disabled there. You'll have to use the data transfer node `lrc-xfer.lbl.gov`:

```shell
$ ssh lrc-xfer.lbl.gov
```

Download and unpack the Scala build tool [sbt](https://www.scala-sbt.org/):

```shell
$ wget https://piccolo.link/sbt-1.3.10.tgz
$ tar xvz sbt-1.3.10.tgz
```

Load the module for JDK 1.8.0:

```shell
$ export MODULEPATH=$MODULEPATH:/global/software/sl-7.x86_64/modfiles/langs
$ module load java
$ java -version
java version "1.8.0_121"
Java(TM) SE Runtime Environment (build 1.8.0_121-b13)
Java HotSpot(TM) 64-Bit Server VM (build 25.121-b13, mixed mode)
```

Clone and build SpaRC:

```shell
$ git clone https://github.com/shawfdong/sparc.git
$ cd sparc
$ cp build.sbt.spark2.1.0 build.sbt
$ ~/sbt/bin/sbt assembly
```

This will create a fat jar file `~/sparc/target/scala-2.11/LocalCluster-assembly-0.2.jar`. Copy it to your Lustre scratch space:

```
$ cp target/scala-2.11/LocalCluster-assembly-0.2.jar /global/scratch/$USER/
```

While you are at it, also download a sample sequence data file:

```console
$ cd /global/scratch/$USER
$ curl http://s3.amazonaws.com/share.jgi-ga.org/sparc_example/illumina_02G_merged_with_id.seq.gz -o sample.seq.gz
$ gunzip sample.seq.gz
$ wc -l sample.seq
6343345 sample.seq
$ head -2 sample.seq
6	HISEQ13:204:C8T6VANXX:1:1101:3726:1992	NATATTCCCGTTCTGATATTGCGTTAAGTCGTTCCCCTAAGCCGGCCCTCCTTATCGAGCGCGCCGGCTTTTTTTGCCATGTTCAGCGAATCACAGGACAAGATACTTCACCTAACGTAGTAGATGGTTCTATGCTTAAGGGCAAGGTGTNTTAATCTCGATATCCGCCTGTTTTAATAAATCAGCGACGAAGCGATGGGAGGATAAGCGCTCGTCAAAAACCACGCGCTTTTTTTCTAAGGTGGGTAAGTTCAAGGTAACACCCCCACTATGCCTATGAGTGAATTGGTAACACCTTGCC
60	HISEQ13:204:C8T6VANXX:1:1101:4370:1919	NCGTGCGCCCATCTCCGTGGCTAAACAGCTTGAGGTGGAAATTCGCCAGTGGATACAGCAGCATGCAGCGACAGGCGGGCGTCGCCTCCCTTCGATACGCCATTTAGCAGCAACACATAACGTCAGCCGCAATGCAGTCATTGAAGCTTANGTAAGGTCTTCTCCTTCGCGCCAATCGTTAGGTAACCAGCCGCAGCCCAGTTTCAATGACTGTTCATCGGTGTTAAACACGCCCCATAAGCCATTCGTCACTTCTTCCAATGGCGTTGATGACGCGGGTTGAACCAGTTTCAGCGCGTTA
```

Now you can exit from `lrc-xfer.lbl.gov`.

*Alternatively*, you could build SpaRC on your local computer, then upload the assembled fat jar file to your Lustre scratch space on Lawrencium.

## Running SpaRC interactively

SSH to a Lawrencium login node:

```shell
$ ssh lrc-login.lbl.gov
```

For demonstration purpose, request 2 nodes from the [lr6 partition](https://sites.google.com/a/lbl.gov/high-performance-computing-services-group/lbnl-supercluster/lawrencium) for the interactive job:

```
$ cd /global/scratch/$USER
$ srun -p lr6 --qos=lr_normal -N 2 -t 1:00:0 --account=<NAME_OF_YOUR_PROJECT_ACCOUNT> --pty bash
```

When the job starts, you'll be given exclusive allocation to 2 compute nodes in the lr6 partition, each one of which has 2x 16-core Intel Xeon Gold 6130 CPUs and 96 GB memory (so in total, there will be 64 cores and 192GB memory available in your Spark cluster). And you'll be dropped to a `bash` shell on one of the compute nodes. Start Spark On Demand (SOD):

```
$ source /global/home/groups/allhands/bin/spark_helper.sh
$ spark-start
```

Run the first Spark job on SOD (you might want to tune the values for `--executor-cores`, `--num-executors` and `--executor-memory`):

```
$ SCRATCH=/global/scratch/$USER
$ JAR=$SCRATCH/LocalCluster-assembly-0.2.jar
$ OPT1="--master $SPARK_URL --executor-cores 4 --num-executors 16 --executor-memory 12g"
$ OPT2="--conf spark.executor.extraClassPath=$JAR \
    --conf spark.driver.maxResultSize=8g \
    --conf spark.network.timeout=360000 \
    --conf spark.speculation=true \
    --conf spark.default.parallelism=100 \
    --conf spark.eventLog.enabled=false"
$ spark-submit $OPT1 $OPT2 \
    $JAR KmerCounting --wait 1 \
    -i $SCRATCH/sample.seq \
    -o $SCRATCH/test_kc_seq_31 --format seq -k 31 -C
```

Run the second Spark job:

```
$ spark-submit $OPT1 $OPT2 \
    $JAR KmerMapReads2 --wait 1 \
    --reads $SCRATCH/sample.seq \
    --format seq -o $SCRATCH/test_kmerreads.txt_31 -k 31 \
    --kmer $SCRATCH/test_kc_seq_31 \
    --contamination 0 --min_kmer_count 2 \
    --max_kmer_count 100000 -C --n_iteration 1
```

Run the third Spark job:

```
$ spark-submit $OPT1 $OPT2 \
    $JAR GraphGen2 --wait 1 \
    -i $SCRATCH/test_kmerreads.txt_31 \
    -o $SCRATCH/test_edges.txt_31 \
    --min_shared_kmers 2 --max_degree 50 -n 1000
```

Run the fourth Spark job:

```
$ spark-submit $OPT1 $OPT2 \
    $JAR GraphLPA2 --wait 1 \
    -i $SCRATCH/test_edges.txt_31 \
    -o $SCRATCH/test_lpa.txt_31 \
    --min_shared_kmers 2 --max_shared_kmers 20000 \
    --min_reads_per_cluster 2 --max_iteration 10 -n 1000
```

Run the fifth Spark job:

```
$ spark-submit $OPT1 $OPT2 \
    $JAR CCAddSeq --wait 1 \
    -i $SCRATCH/test_lpa.txt_31 \
    --reads $SCRATCH/sample.seq \
    -o $SCRATCH/sample_lpaseq.txt_31
```

Once you are done, don't forget to stop Spark On Demand and exit from the interactive job: 

```
$ spark-stop
$ exit
```

## Running SpaRC in batch mode

To run SpaRC in batch mode, write a Slurm job script and call it `sparc.slurm`:

```bash
#!/bin/bash
#SBATCH --job-name=sparc
#SBATCH --partition=lr6
#SBATCH --qos=lr_normal
#SBATCH --account=<NAME_OF_YOUR_PROJECT_ACCOUNT>
#SBATCH --nodes=2
#SBATCH --time=01:00:00

source /global/home/groups/allhands/bin/spark_helper.sh
# Start Spark On Demand
spark-start

SCRATCH=/global/scratch/$USER
JAR=$SCRATCH/LocalCluster-assembly-0.2.jar
OPT1="--master $SPARK_URL --executor-cores 4 --num-executors 16 --executor-memory 12g"
OPT2="--conf spark.executor.extraClassPath=$JAR \
    --conf spark.driver.maxResultSize=8g \
    --conf spark.network.timeout=360000 \
    --conf spark.speculation=true \
    --conf spark.default.parallelism=100 \
    --conf spark.eventLog.enabled=false"

# 1st Spark job
spark-submit $OPT1 $OPT2 \
    $JAR KmerCounting --wait 1 \
    -i $SCRATCH/sample.seq \
    -o $SCRATCH/test_kc_seq_31 --format seq -k 31 -C

# 2nd Spark job
spark-submit $OPT1 $OPT2 \
    $JAR KmerMapReads2 --wait 1 \
    --reads $SCRATCH/sample.seq \
    --format seq -o $SCRATCH/test_kmerreads.txt_31 -k 31 \
    --kmer $SCRATCH/test_kc_seq_31 \
    --contamination 0 --min_kmer_count 2 \
    --max_kmer_count 100000 -C --n_iteration 1

# 3rd Spark job
spark-submit $OPT1 $OPT2 \
    $JAR GraphGen2 --wait 1 \
    -i $SCRATCH/test_kmerreads.txt_31 \
    -o $SCRATCH/test_edges.txt_31 \
    --min_shared_kmers 2 --max_degree 50 -n 1000
Run the fourth Spark job:

# 4th Spark job
spark-submit $OPT1 $OPT2 \
    $JAR GraphLPA2 --wait 1 \
    -i $SCRATCH/test_edges.txt_31 \
    -o $SCRATCH/test_lpa.txt_31 \
    --min_shared_kmers 2 --max_shared_kmers 20000 \
    --min_reads_per_cluster 2 --max_iteration 10 -n 1000

# 5th Spark job
spark-submit $OPT1 $OPT2 \
    $JAR CCAddSeq --wait 1 \
    -i $SCRATCH/test_lpa.txt_31 \
    --reads $SCRATCH/sample.seq \
    -o $SCRATCH/sample_lpaseq.txt_31

# Stop Spark On Demand
spark-stop
```

Then submit the job with:

```shell
sbatch sparc.slurm
```

## Known issues and future improvements

1. Presumably, there is a switch `-i` to `spark-start` that would enable communications over the IPoIB network. But it doesn't work!
2. Spark 2.1.0 is a bit old.

## References

1. <https://academic.oup.com/bioinformatics/article/35/5/760/5078476>
2. <https://peerj.com/articles/8966/>