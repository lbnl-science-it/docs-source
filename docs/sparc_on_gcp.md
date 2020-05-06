# Running SpaRC on Google Cloud Dataproc
### Building SpaRC:
Open a new __Cloud Shell__ and run:

```shell
git clone https://github.com/shawfdong/sparc.git
cd sparc
wget https://piccolo.link/sbt-1.3.10.tgz
tar xvz sbt-1.3.10.tgz
./sbt/bin/sbt assembly
```
A jar file should be created at: `target/scala-2.11/LocalCluster-assembly-0.2.jar`

### Upload Data to Google Cloud Storage
1. Navigate to __Storage__ and select __Storage > Browser__.

2. Click __Create Bucket__.
`
3. Specify __your project name__ as the __bucket name__.

4. Click __Create__.

5. Copy the compiled SpaRC `LocalCluster-assembly-0.2.jar` and a sample input file `sample_small.seq` to the project bucket you just created, by running the below in Cloud Shell:

```shell
gsutil cp target/scala-2.11/LocalCluster-assembly-0.2.jar gs://$DEVSHELL_PROJECT_ID
cd data/small
cp sample.seq sample_small.seq
gsutil cp sample_small.seq gs://$DEVSHELL_PROJECT_ID
```
### [Launch Dataproc](https://cloud.google.com/dataproc)

### Run SpaRC job on Dataproc
1. In the __Dataproc__ console, click __Jobs__.

2. Click __Submit job__.

3. For __Job type__, select __Spark__; for __Main class or jar__ and __Jar files__, specify the location of the SpaRC jar file you uploaded to your bucket. Your __bucket-name__ is __your project name__: `gs://<bucket-name>/LocalCluster-assembly-0.2.jar`. 
  
   For __Arguments__, enter each of these arguments separately:
   ```json
       "args": [
                "KmerCounting",
                "--input",
                "gs://<my-project-ID>/sample_small.seq",
                "--output",
                "tset.log",
                "--kmer_length",
                "31"
         ]
   ```

   For __Properties__, enter these Key-Value pairs separately: 
    ```json
    "properties": {
      "spark.executor.extraClassPath": "gs://<my-project-ID>/LocalCluster-assembly-0.2.jar",
      "spark.driver.maxResultSize": "8g",
      "spark.network.timeout": "360000",
      "spark.default.parallelism": "4",
      "spark.eventLog.enabled": "false"
    }
    ```
 
 4. Click __Submit__
