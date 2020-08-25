## Overview:
There are a number of options available for backing up your data. 

For a lab IT supported option, you can use the Commvault service offered by UC Berkeley [link_to commvault_info](link_me_here).

If you would rather setup and manage your own backup processes, there are other options available.  You'll need to setup and configure a backup client on your system to manage the transfer of the files, and you'll need a data server setup to store the backup copies of your files.

Science IT is available to help you select your best backup option.  Email [scienceit@lbl.gov](mailto:scienceit@lbl.gov) to schedule a consultation.

Backup Solutions

| __Tool__            | __Commvault__ | __Rclone__  | __Duplicati__ | __MSP360__ |
| -------------       | ----------    | ----------  | ------------- | -----------|
|                     | UC Managed    | Open Source | Open Source   | Commercial |
| Lab IT Supported    | Y             | N           | N             | N          |
| __Runs on:__        | Y             |             |               |            |
| Windows             | Y             | Y           | Y             | Y          |
| MacOS               | Y             | Y           | Y             | Y          |
| Linux               | Y             | Y           | Y             | Y          |
| __Stores data to:__ | Y             |             |               |            |
| UC Berkeley         | Y             | N           | N             | N          |
| Google Drive        | N             | Y           | Y             | N          |
| Amazon S3           | N             | Y           | Y             | Y          |
| Google Cloud        | N             | Y           | Y             | Y          |
| [SPSS](https://commons.lbl.gov/pages/viewpage.action?pageId=184100826) | N | Y | Y | N |

## Data storage location options:

### UC Berkeley Commvault

Monthly costs for data is $0.28/GB/month and can be charged to your project ID via recharge.  There are no extra charges for restoring data, and restores are available 24x7 via a self-serve web interface.

### Google Drive

All LBL users have free and unlimited data storage available to them in Google Drive.  For a simple backup process, this might be your best option.  

### Amazon S3 or Google Cloud

Generally speaking, Amazon S3 or Google Cloud storage will cost about $0.023/GB/month for data storage.  Additionally, there can be egress charges depending on your need to restore data back to your system.  Science IT can get you setup on the LBL master payer program for either/both services.

### [SPSS](https://commons.lbl.gov/pages/viewpage.action?pageId=184100826)

SPSS is a highly available and scalable storage service offered by the IT division as part of the Science IT initiative.  SPSS affords lab researchers a file storage platform to support data workflows without the cost of dedicated hardware or the administration burden of running their own IT infrastructure.  SPSS will cost about $0.013/GB/month.

## Open source backup solutions:

### Rclone

Rclone is a command line program to manage files on cloud storage and sync files and directories to and from: Google Drive, Amazon S3, Dropbox, Backblaze B2, OneDrive, Swift, Hubic, Cloudfiles, Amazon Drive, Google Cloud Storage, Yandex Files, the local filesystem and more. Rclone is widely used on Linux, Windows and Mac. Third party developers create innovative backup, restore, GUI and business process solutions using the rclone command line or API.

#### [Install](https://rclone.org/downloads/)
#### [Configuration](https://rclone.org/docs/#configure)
#### Documentation
   * _Amazon S3_: [https://rclone.org/s3/](https://rclone.org/s3/)
   * _Google Drive_: [https://rclone.org/drive/](https://rclone.org/drive/)
   * _Google Cloud Storage_: [https://rclone.org/googlecloudstorage/](https://rclone.org/googlecloudstorage/)

### Duplicati
Duplicati is a backup client that securely stores encrypted, incremental, compressed remote backups of local files on cloud storage services and remote file servers. Duplicati supports not only various online backup services like OneDrive, Amazon S3, Backblaze, Rackspace Cloud Files, Tahoe LAFS, and Google Drive, but also any servers that support SSH/SFTP, WebDAV, or FTP. Duplicati uses standard components such as rdiff, zip, AESCrypt, and GnuPG. This allows users to recover backup files even if Duplicati is not available. Released under the terms of the GNU Lesser General Public License (LGPL), Duplicati is free software.

#### [Install](https://www.duplicati.com/download)
####  Configuration
   * [GUI](https://duplicati.readthedocs.io/en/latest/03-using-the-graphical-user-interface/)
   * [Command-Line](https://duplicati.readthedocs.io/en/latest/04-using-duplicati-from-the-command-line/)
#### Documentation
   * _Amazon S3_: [https://duplicati.readthedocs.io/en/latest/05-storage-providers/#s3-compatible](https://duplicati.readthedocs.io/en/latest/05-storage-providers/#s3-compatible)
   * _Google Drive_: [https://duplicati.readthedocs.io/en/latest/05-storage-providers/#google-drive](https://duplicati.readthedocs.io/en/latest/05-storage-providers/#google-drive)
   * _Google Cloud Storage_: [https://duplicati.readthedocs.io/en/latest/05-storage-providers/#google-cloud-storage](https://duplicati.readthedocs.io/en/latest/05-storage-providers/#google-cloud-storage)

## Commercial backup solutions: 

### MSP360
MSP360™ Backup for Amazon S3 and Google Cloud allows you to backup files, folders and system image on Windows, Mac or Linux to Amazon S3 and Google Cloud. MSP360™ Backup features flexible scheduling options, 256-bit AES encryption, backup consistency check and more.

* _Amazon S3_: [https://www.msp360.com/backup/amazon-s3.aspx](https://www.msp360.com/backup/amazon-s3.aspx)
* _Google Cloud Storage_: [https://www.msp360.com/backup/google-cloud.aspx](https://www.msp360.com/backup/google-cloud.aspx)

## References:
* [https://rclone.org/commands/](https://rclone.org/commands/)
* [https://duplicati.readthedocs.io/en/latest/](https://duplicati.readthedocs.io/en/latest/)
* [https://www.msp360.com/backup.aspx](https://www.msp360.com/backup.aspx)
