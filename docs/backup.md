## Overview:
|               | Open Source | Windows | macOS | Linux | Amazon S3 | Google Drive | Google Cloud |
|---------------|-------------|---------|-------|-------|-----------|--------------|--------------|
| __Rclone__    | Y           | Y       | Y     | Y     | Y         | Y            | Y            |
| __Duplicati__ | Y           | Y       | Y     | Y     | Y         | Y            | Y            |
| __MSP360__    | N           | Y       | Y     | Y     | Y         | N            | Y            |
| __Veeam__     | N           | Y       | N     | N     | Y         | N            | N            |


## Open source backup solutions:

### Rclone

Rclone is a command line program to manage files on cloud storage and sync files and directories to and from: Google Drive, Amazon S3, Dropbox, Backblaze B2, OneDrive, Swift, Hubic, Cloudfiles, Amazon Drive, Google Cloud Storage, Yandex Files, the local filesystem and more. Rclone is widely used on Linux, Windows and Mac. Third party developers create innovative backup, restore, GUI and business process solutions using the rclone command line or API.

#### [Install](https://rclone.org/downloads/)
#### [Configuration](https://rclone.org/docs/#configure)
#### Documentation:
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

### Veeam
Veeam® Backup & Replication™ delivers Availability for all your cloud, virtual and physical workloads. Through a simple-by-design management console, you can easily achieve fast, flexible and reliable backup, recovery and replication for all your applications and data. https://www.veeam.com/vm-backup-recovery-replication-software.html

* _Amazon S3_: [https://helpcenter.veeam.com/docs/backup/hyperv/adding_amazon_object_storage.html?ver=100](https://helpcenter.veeam.com/docs/backup/hyperv/adding_amazon_object_storage.html?ver=100)

## References:
* [https://rclone.org/commands/](https://rclone.org/commands/)
* [https://duplicati.readthedocs.io/en/latest/](https://duplicati.readthedocs.io/en/latest/)
* [https://www.msp360.com/backup.aspx](https://www.msp360.com/backup.aspx)
* [https://helpcenter.veeam.com/docs/backup/vsphere/system_requirements.html?ver=100](https://helpcenter.veeam.com/docs/backup/vsphere/system_requirements.html?ver=100)
