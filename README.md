# Deteksi Objek menggunakan YOLOv8


> Penelitian telah dilakukan untuk menyelesaikan isu pengendara motor yang tidak mengenakan helm, untuk mendeteksi apakah seorang pengendara sepeda motor memakai helm atau tidak. Salah satu sistem yang telah dikembangkan adalah Roboflow yang memanfaatkan kecerdasan buatan dan mengimplementasikan algoritma YOLOv8

Anggota Kelompok:
1.   Ferrari Ahmad A.				(15220247)  
2. Reza Dwi Ananda	 	   (15220287)  
3. Ridho Darmawan Z. 			(15220154)
4. Teuku Vaickal R.I			(15220658)

Dependensi yang akan kita gunakan:


*   Ultralytics
*   zipfile
*   cv2
*   matplotlib



Total Citra:

```
Total Images: 208
Total Train: 145
Total Valid: 42
Total Test: 21

```


Struktur File:
```
📦HelmYOLOv8
 ┣ 📂test
 ┃ ┣ 📂images
 ┃ ┃ ┣ 📜BikesHelmets292_png.rf.8c5c58e0375231b6a7ed1d5fe1a9783b.jpg
 ┃ ┃ ┣ 📜BikesHelmets60_png.rf.9bc644dd3535900feb1c8be251c62b3b.jpg
 ┃ ┃ ┣ 📜BikesHelmets650_png.rf.8a87745356003e78f5ae67b387114287.jpg
 ┃ ┃ ┣ 📜BikesHelmets748_png.rf.e715e370945d3b13540072f9b6081c0f.jpg
 ┃ ┃ ┣ 📜Image-25-_jpg.rf.4eef3f6a0eaac81040eef1278d30c4dc.jpg
 ┃ ┃ ┣ 📜Image-29-_jpg.rf.c6c09c66a961b7afce0bb42aa32b5c3b.jpg
 ┃ ┃ ┣ 📜Image-40-_jpg.rf.f4c274484f10bb11a0289c54901e2314.jpg
 ┃ ┃ ┣ 📜Image-41-_jpeg.rf.3dc1bd67504d50f4a33a89b8e69a5862.jpg
 ┃ ┃ ┣ 📜Image-43-_jpeg.rf.a13c15701a2ba677f010e47654ef8aab.jpg
 ┃ ┃ ┣ 📜Image-47-_jpg.rf.1b5068751b2962084ede11f3000d9c02.jpg
 ┃ ┃ ┣ 📜Image-48-_jpeg.rf.417463794c7caac38f793aed6b303525.jpg
 ┃ ┃ ┣ 📜Image-68-_jpeg.rf.132dd4ca76937f11a3cd19032bcc16ae.jpg
 ┃ ┃ ┣ 📜Image-7-_jpeg.rf.eeb49c758534128c826a92f3271e6923.jpg
 ┃ ┃ ┣ 📜Image-94-_jpg.rf.58c6a72fb3acde83e4890d25a6da4639.jpg
 ┃ ┃ ┣ 📜Image-95-_jpg.rf.04fdc0093b42cf9c027a7ed026c8de9d.jpg
 ┃ ┃ ┣ 📜nohelmet-1-_jpg.rf.14b29fa06fc2f2c5a02c2e4c0cd2997e.jpg
 ┃ ┃ ┣ 📜nohelmet-11-_png.rf.e48442f9da4b2a72976ed8489d279b49.jpg
 ┃ ┃ ┣ 📜nohelmet-19-_png.rf.846ccd92c6f48292ea605aa8dfb9539c.jpg
 ┃ ┃ ┣ 📜nohelmet-28-_jpg.rf.cd70d30d306daecf7b1c393474e45405.jpg
 ┃ ┃ ┣ 📜nohelmet-42-_jpg.rf.c5534a3560159e9903b3458bd8822fa6.jpg
 ┃ ┃ ┗ 📜nohelmet-77-_jpg.rf.a4674d411bc719bb3d35831aa6a1bfec.jpg
 ┃ ┗ 📂labels
 ┃ ┃ ┣ 📜BikesHelmets292_png.rf.8c5c58e0375231b6a7ed1d5fe1a9783b.txt
 ┃ ┃ ┣ 📜BikesHelmets60_png.rf.9bc644dd3535900feb1c8be251c62b3b.txt
 ┃ ┃ ┣ 📜BikesHelmets650_png.rf.8a87745356003e78f5ae67b387114287.txt
 ┃ ┃ ┣ 📜BikesHelmets748_png.rf.e715e370945d3b13540072f9b6081c0f.txt
 ┃ ┃ ┣ 📜Image-25-_jpg.rf.4eef3f6a0eaac81040eef1278d30c4dc.txt
 ┃ ┃ ┣ 📜Image-29-_jpg.rf.c6c09c66a961b7afce0bb42aa32b5c3b.txt
 ┃ ┃ ┣ 📜Image-40-_jpg.rf.f4c274484f10bb11a0289c54901e2314.txt
 ┃ ┃ ┣ 📜Image-41-_jpeg.rf.3dc1bd67504d50f4a33a89b8e69a5862.txt
 ┃ ┃ ┣ 📜Image-43-_jpeg.rf.a13c15701a2ba677f010e47654ef8aab.txt
 ┃ ┃ ┣ 📜Image-47-_jpg.rf.1b5068751b2962084ede11f3000d9c02.txt
 ┃ ┃ ┣ 📜Image-48-_jpeg.rf.417463794c7caac38f793aed6b303525.txt
 ┃ ┃ ┣ 📜Image-68-_jpeg.rf.132dd4ca76937f11a3cd19032bcc16ae.txt
 ┃ ┃ ┣ 📜Image-7-_jpeg.rf.eeb49c758534128c826a92f3271e6923.txt
 ┃ ┃ ┣ 📜Image-94-_jpg.rf.58c6a72fb3acde83e4890d25a6da4639.txt
 ┃ ┃ ┣ 📜Image-95-_jpg.rf.04fdc0093b42cf9c027a7ed026c8de9d.txt
 ┃ ┃ ┣ 📜nohelmet-1-_jpg.rf.14b29fa06fc2f2c5a02c2e4c0cd2997e.txt
 ┃ ┃ ┣ 📜nohelmet-11-_png.rf.e48442f9da4b2a72976ed8489d279b49.txt
 ┃ ┃ ┣ 📜nohelmet-19-_png.rf.846ccd92c6f48292ea605aa8dfb9539c.txt
 ┃ ┃ ┣ 📜nohelmet-28-_jpg.rf.cd70d30d306daecf7b1c393474e45405.txt
 ┃ ┃ ┣ 📜nohelmet-42-_jpg.rf.c5534a3560159e9903b3458bd8822fa6.txt
 ┃ ┃ ┗ 📜nohelmet-77-_jpg.rf.a4674d411bc719bb3d35831aa6a1bfec.txt
 ┣ 📂train
 ┃ ┣ 📂images
 ┃ ┃ ┣ 📜160610_jpg.rf.cf9ee04e91810899d842ef5feed9b700.jpg
 ┃ ┃ ┣ 📜160613_jpg.rf.ace7457f0edabb2f94213250858579eb.jpg
 ┃ ┃ ┣ 📜BikesHelmets13_png.rf.44b2421206bd428af238fa10cfc772f7.jpg
 ┃ ┃ ┣ 📜BikesHelmets158_png.rf.e6879a189a98504a7ffb11a503725826.jpg
 ┃ ┃ ┣ 📜BikesHelmets167_png.rf.e2de52c0b6346d1029b2f547c0bc7f7b.jpg
 ┃ ┃ ┣ 📜BikesHelmets209_png.rf.bf3175b9a9ca43563d7d4bbe9a09f120.jpg
 ┃ ┃ ┣ 📜BikesHelmets282_png.rf.d22efa640f3e6559093004497c4caa71.jpg
 ┃ ┃ ┣ 📜BikesHelmets288_png.rf.04822f17bb0437ca3728a3eb154c6f4f.jpg
 ┃ ┃ ┣ 📜BikesHelmets380_png.rf.7719b999f8dd72ed75ea156f946c1425.jpg
 ┃ ┃ ┣ 📜BikesHelmets396_png.rf.48b485293de26b1c708d7b1ec9e0ae08.jpg
 ┃ ┃ ┣ 📜BikesHelmets42_png.rf.7db6a6611e8fabc7e3beecedded42a9b.jpg
 ┃ ┃ ┣ 📜BikesHelmets434_png.rf.a4a10cbb3e49fa97b729293e27cd89a7.jpg
 ┃ ┃ ┣ 📜BikesHelmets451_png.rf.ce48ff6aff697f98a734b0b8b1c49152.jpg
 ┃ ┃ ┣ 📜BikesHelmets468_png.rf.8297f31b73a89ca64f38d03d0181a2d8.jpg
 ┃ ┃ ┣ 📜BikesHelmets476_png.rf.f76ae5571276a493ce10c4aab0aeaec6.jpg
 ┃ ┃ ┣ 📜BikesHelmets508_png.rf.1aea1a9ec6ac416f88a8a0bed5824ed3.jpg
 ┃ ┃ ┣ 📜BikesHelmets53_png.rf.3b92d7f1ff7352165ba6358df95b1a99.jpg
 ┃ ┃ ┣ 📜BikesHelmets549_png.rf.b978a261301ebb860b8ce11c2e3ff96b.jpg
 ┃ ┃ ┣ 📜BikesHelmets580_png.rf.50a0727292eeee9ac2cb07a68738b874.jpg
 ┃ ┃ ┣ 📜BikesHelmets619_png.rf.2ca88fe2c43cc7c4a07c2f39a2e62ed5.jpg
 ┃ ┃ ┣ 📜BikesHelmets629_png.rf.0033ba35c431e0e697c873117b6ee1de.jpg
 ┃ ┃ ┣ 📜BikesHelmets653_png.rf.1b87c0477ca16f3941c2f94b87ebdd29.jpg
 ┃ ┃ ┣ 📜BikesHelmets721_png.rf.30810b07a4e959aa4afe5f02831b790b.jpg
 ┃ ┃ ┣ 📜BikesHelmets90_png.rf.bf5fc264672ae7df292081516d54e6d9.jpg
 ┃ ┃ ┣ 📜BikesHelmets92_png.rf.72a5228e7ebda6d64e8f3df00ae845d9.jpg
 ┃ ┃ ┣ 📜Image-22-_jpg.rf.3950dd59f63b678be71d230f485812c8.jpg
 ┃ ┃ ┣ 📜Image-23-_jpeg.rf.1460483cbaa9346c580fd25eb444ec0b.jpg
 ┃ ┃ ┣ 📜Image-24-_jpeg.rf.70d0ae0bf08db38a07d388e1f12da8cb.jpg
 ┃ ┃ ┣ 📜Image-24-_jpg.rf.86fd0c6716364978fd27b34b9e2f48b5.jpg
 ┃ ┃ ┣ 📜Image-25-_jpeg.rf.7427dcf5202a9129e74e71cfc7d81588.jpg
 ┃ ┃ ┣ 📜Image-26-_jpeg.rf.57cebecc105b699d8f537f4e3cfa017c.jpg
 ┃ ┃ ┣ 📜Image-28-_jpeg.rf.42c1a872709de30b8ba46b955428d254.jpg
 ┃ ┃ ┣ 📜Image-28-_jpg.rf.c6a31f629bf5ace23b83d5b4cad20818.jpg
 ┃ ┃ ┣ 📜Image-3-_jpg.rf.30e0eaa423b29ed9d64ccf30a304bf35.jpg
 ┃ ┃ ┣ 📜Image-3-_png.rf.bf2fc1568d009618eed1832b7b0a8e8b.jpg
 ┃ ┃ ┣ 📜Image-30-_jpeg.rf.64db4de98e8aac3779911d265eafb2dd.jpg
 ┃ ┃ ┣ 📜Image-31-_jpeg.rf.44cc8028dc0e22886bc4ac5fa1422df3.jpg
 ┃ ┃ ┣ 📜Image-33-_jpeg.rf.16768fa6fa82f91a112369c7a01ee2ec.jpg
 ┃ ┃ ┣ 📜Image-34-_jpeg.rf.e53cdfc91b4e7fb3ea6e981309107107.jpg
 ┃ ┃ ┣ 📜Image-39-_jpeg.rf.9811ac17db0b82c447a355079ce67977.jpg
 ┃ ┃ ┣ 📜Image-39-_jpg.rf.63ce16d386e22b4eb09830a5661cdc59.jpg
 ┃ ┃ ┣ 📜Image-4-_jpeg.rf.3309204fa4e6282af53e79c67c0b50a6.jpg
 ┃ ┃ ┣ 📜Image-4-_jpg.rf.d5a26aa7c03daae5aca8641b936a5ab5.jpg
 ┃ ┃ ┣ 📜Image-42-_jpeg.rf.fb2f054d1cf35460819e33d6aa4510f9.jpg
 ┃ ┃ ┣ 📜Image-43-_jpg.rf.0846568e9dd34d32f687e9a35d560e63.jpg
 ┃ ┃ ┣ 📜Image-44-_jpeg.rf.35641e5218f749921ea59b8fc69740ce.jpg
 ┃ ┃ ┣ 📜Image-46-_jpeg.rf.09eff07d88ebb622a7da6ad484cf5648.jpg
 ┃ ┃ ┣ 📜Image-46-_jpg.rf.2f8f2dcf14ac897a0db0bc0c13a896ca.jpg
 ┃ ┃ ┣ 📜Image-47-_jpeg.rf.693adfa22f8604d19a959e23a450dc63.jpg
 ┃ ┃ ┣ 📜Image-5-_jpeg.rf.b61aa14d08f2793979266e4758914161.jpg
 ┃ ┃ ┣ 📜Image-5-_jpg.rf.9bebf09d483c171633eedbc003140745.jpg
 ┃ ┃ ┣ 📜Image-50-_jpeg.rf.ec83f338f66fb6e69eda8e835404aa58.jpg
 ┃ ┃ ┣ 📜Image-50-_jpg.rf.597bd890ba054ea0e74273cf57660eaa.jpg
 ┃ ┃ ┣ 📜Image-51-_jpeg.rf.1da770c5d99d945e0d9a0f4460ac8c38.jpg
 ┃ ┃ ┣ 📜Image-51-_jpg.rf.43bdc8b7250612c05e5aef6a3360a289.jpg
 ┃ ┃ ┣ 📜Image-52-_jpeg.rf.7db136f49a216ec7448512bd8be3f85d.jpg
 ┃ ┃ ┣ 📜Image-53-_jpeg.rf.45fa43aa0d6400dd902b55b9a8ad1980.jpg
 ┃ ┃ ┣ 📜Image-54-_jpg.rf.3ecca0b2ce8036eb4c35cf542eb51b0e.jpg
 ┃ ┃ ┣ 📜Image-56-_jpg.rf.4672163b5385a3b1596f1b0459c46cd1.jpg
 ┃ ┃ ┣ 📜Image-6-_jpg.rf.a55c109ffddc87038d96653295fc4548.jpg
 ┃ ┃ ┣ 📜Image-67-_jpg.rf.997b17a7a88353a774c20f6c5117b1be.jpg
 ┃ ┃ ┣ 📜Image-68-_jpg.rf.634d2db1584203d680cf0d56d7f14d5e.jpg
 ┃ ┃ ┣ 📜Image-69-_jpeg.rf.f54b2e051d5f298f1c54cead8f8fcb3b.jpg
 ┃ ┃ ┣ 📜Image-70-_jpeg.rf.b5e1582df147426fc7d2755497b9f69e.jpg
 ┃ ┃ ┣ 📜Image-70-_jpg.rf.2c656240f4f61b6478780a7cd7f04c27.jpg
 ┃ ┃ ┣ 📜Image-71-_jpeg.rf.26d2fc54f6255608d99af76c700e157c.jpg
 ┃ ┃ ┣ 📜Image-71-_jpg.rf.68f6ff884ccdd03e241d9179d10c6fe3.jpg
 ┃ ┃ ┣ 📜Image-75-_jpg.rf.6e0fe9e2d49ef119018648dffad33f7e.jpg
 ┃ ┃ ┣ 📜Image-8-_jpeg.rf.c86f41fcf180e9139f5a3b1fb11e9502.jpg
 ┃ ┃ ┣ 📜Image-8-_jpg.rf.3a3fc1af44ac924651b5c0689949633f.jpg
 ┃ ┃ ┣ 📜Image-9-_jpeg.rf.2f2377133a3912d1692d3620c6dc9664.jpg
 ┃ ┃ ┣ 📜Image-97-_jpg.rf.e6ab3c968335ff83217e6fda2ab2c3aa.jpg
 ┃ ┃ ┣ 📜nohelmet-10-_jpg.rf.3a723ac459368743f8a0b52f606ad412.jpg
 ┃ ┃ ┣ 📜nohelmet-11-_jpg.rf.41537c8a25c3282b89701a7e521e74e2.jpg
 ┃ ┃ ┣ 📜nohelmet-12-_jpg.rf.5ef447ee7a06a714b93301e7f4cfa092.jpg
 ┃ ┃ ┣ 📜nohelmet-12-_png.rf.dd5d9c21b7379c603e15a9c0c418549b.jpg
 ┃ ┃ ┣ 📜nohelmet-14-_jpg.rf.0b62e37e25c9896ae08eb1f7bd897e53.jpg
 ┃ ┃ ┣ 📜nohelmet-14-_png.rf.16edb4bc5d0e16e85c80fca0390ddb6b.jpg
 ┃ ┃ ┣ 📜nohelmet-15-_jpg.rf.9ff2467c17712b7f2d6ee93f0e05684b.jpg
 ┃ ┃ ┣ 📜nohelmet-15-_png.rf.09d42d04dc0e988d51ed0eb9140efd89.jpg
 ┃ ┃ ┣ 📜nohelmet-16-_jpg.rf.6f64dc830197fe4c5a34618cc310ffe5.jpg
 ┃ ┃ ┣ 📜nohelmet-16-_png.rf.3c82164a6f786c59d9b9c94cd3aa213d.jpg
 ┃ ┃ ┣ 📜nohelmet-17-_jpg.rf.765341a733365aadda335913b3ed83e1.jpg
 ┃ ┃ ┣ 📜nohelmet-17-_png.rf.65f35c64920748129ea4a1d54cd59c99.jpg
 ┃ ┃ ┣ 📜nohelmet-18-_jpg.rf.ec3ec65681de1eff87395420ce8c28c6.jpg
 ┃ ┃ ┣ 📜nohelmet-18-_png.rf.54b257eb62c985af87fb62ba22819167.jpg
 ┃ ┃ ┣ 📜nohelmet-19-_jpg.rf.bd773640223bfbc432ee56a7d7d512ab.jpg
 ┃ ┃ ┣ 📜nohelmet-2-_jpg.rf.9890ce593237f65dd2c8555b170575aa.jpg
 ┃ ┃ ┣ 📜nohelmet-20-_jpg.rf.65eb6247943d0eb4e8df332fbfff561a.jpg
 ┃ ┃ ┣ 📜nohelmet-20-_png.rf.2a7300e1e8f517836115262a7620c098.jpg
 ┃ ┃ ┣ 📜nohelmet-21-_jpg.rf.a4185b59bbf04fbc0887b42ca945ceea.jpg
 ┃ ┃ ┣ 📜nohelmet-21-_png.rf.d0fa33d0915b88b926eb09ae847fd387.jpg
 ┃ ┃ ┣ 📜nohelmet-22-_jpg.rf.9fb61668ecc5cb7132062efc77249a37.jpg
 ┃ ┃ ┣ 📜nohelmet-22-_png.rf.94a8ebbc5fcb2e55fe94a412a44c43f8.jpg
 ┃ ┃ ┣ 📜nohelmet-23-_jpg.rf.999079860d064d3ed890a066703e3fc5.jpg
 ┃ ┃ ┣ 📜nohelmet-23-_png.rf.27a616fe48e3ba3952f149086daf3557.jpg
 ┃ ┃ ┣ 📜nohelmet-24-_jpg.rf.69df39b5653c5e3cbec43107c3b8f3f5.jpg
 ┃ ┃ ┣ 📜nohelmet-27-_jpg.rf.00989843a1285a82018dfc4773668301.jpg
 ┃ ┃ ┣ 📜nohelmet-29-_jpg.rf.ce0b90b0476f2d827e06fd34ca34e201.jpg
 ┃ ┃ ┣ 📜nohelmet-3-_jpg.rf.f9214eb29646840300ceb59f50256bdb.jpg
 ┃ ┃ ┣ 📜nohelmet-3-_png.rf.d4ba340fe7c4656878dc4d1e23698871.jpg
 ┃ ┃ ┣ 📜nohelmet-32-_jpg.rf.b9f59c0e1e87a07362f3420b780d62d0.jpg
 ┃ ┃ ┣ 📜nohelmet-33-_jpg.rf.95ff09f84f64e40c324877f8e17e1729.jpg
 ┃ ┃ ┣ 📜nohelmet-34-_jpg.rf.cd54689dccc22b07bb7a89e3f63b2fb6.jpg
 ┃ ┃ ┣ 📜nohelmet-35-_jpg.rf.72ba77af56bee5e3b856ba596ae0fe24.jpg
 ┃ ┃ ┣ 📜nohelmet-36-_jpg.rf.6fd05e137305e715cab13ca933a20d7b.jpg
 ┃ ┃ ┣ 📜nohelmet-37-_jpg.rf.94222a991d6ce0a17a6ded66263f5011.jpg
 ┃ ┃ ┣ 📜nohelmet-38-_jpg.rf.0899b0cad8995377ad1749c891e56dce.jpg
 ┃ ┃ ┣ 📜nohelmet-39-_jpg.rf.9a11e99c61bbc236fd7c5fa36eac74c2.jpg
 ┃ ┃ ┣ 📜nohelmet-4-_jpg.rf.1e2311c18fcfb0018d5bbf78d295512b.jpg
 ┃ ┃ ┣ 📜nohelmet-4-_png.rf.8d63e3f04ded26426881a88c5a7f58e3.jpg
 ┃ ┃ ┣ 📜nohelmet-40-_jpg.rf.8fbaa10454cb4ad4f1523796d24de3b0.jpg
 ┃ ┃ ┣ 📜nohelmet-41-_jpg.rf.b6fbf9a011cf755e06a16656b95ff322.jpg
 ┃ ┃ ┣ 📜nohelmet-44-_jpg.rf.d76174e99799321a92db2fef5975be54.jpg
 ┃ ┃ ┣ 📜nohelmet-46-_jpg.rf.f142ef85cb270231e27f1393ade25770.jpg
 ┃ ┃ ┣ 📜nohelmet-47-_jpg.rf.7de1809902b953a566915f77e6253433.jpg
 ┃ ┃ ┣ 📜nohelmet-48-_jpg.rf.97e882e71117ab79c03421ac7905f956.jpg
 ┃ ┃ ┣ 📜nohelmet-49-_jpg.rf.b1ce16b64180031b0e7e9362a229634f.jpg
 ┃ ┃ ┣ 📜nohelmet-5-_png.rf.2ebb10c1a365886e7fda8eb2646117c5.jpg
 ┃ ┃ ┣ 📜nohelmet-50-_jpg.rf.1e92fc16ec7278da0621df89000d073f.jpg
 ┃ ┃ ┣ 📜nohelmet-52-_jpg.rf.6a1cb9bcfd8c31e500fea014374a36c9.jpg
 ┃ ┃ ┣ 📜nohelmet-53-_jpg.rf.1a467d900fe5523104e45550a07f0d81.jpg
 ┃ ┃ ┣ 📜nohelmet-55-_jpg.rf.8998ffec8a8f4b9fcf49ce583ce803aa.jpg
 ┃ ┃ ┣ 📜nohelmet-56-_jpg.rf.3e6901961565139709393d4661815093.jpg
 ┃ ┃ ┣ 📜nohelmet-57-_jpg.rf.241e7f8293d39be96d1499fc2165329b.jpg
 ┃ ┃ ┣ 📜nohelmet-58-_jpg.rf.5eff0c8d81b380b5bf426f30fe7ebdfb.jpg
 ┃ ┃ ┣ 📜nohelmet-6-_jpg.rf.3526de88d11496f28f459986be3167f8.jpg
 ┃ ┃ ┣ 📜nohelmet-6-_png.rf.7972e4037e28e690025595435decee86.jpg
 ┃ ┃ ┣ 📜nohelmet-60-_jpg.rf.7b7b0c87628ffe503695d489d9a29c7a.jpg
 ┃ ┃ ┣ 📜nohelmet-61-_jpg.rf.fa957673f8d5d3779bbf1888e53f6276.jpg
 ┃ ┃ ┣ 📜nohelmet-62-_jpg.rf.f95fb517f822743ace2c2fad5c4898cc.jpg
 ┃ ┃ ┣ 📜nohelmet-65-_jpg.rf.d0da42d692e8f12f09e96b9e58b60433.jpg
 ┃ ┃ ┣ 📜nohelmet-66-_jpg.rf.6a0e667c76beb7b957d03d370029e3c5.jpg
 ┃ ┃ ┣ 📜nohelmet-68-_jpg.rf.30cda3c26f643cc4027b74a1cf541648.jpg
 ┃ ┃ ┣ 📜nohelmet-69-_jpg.rf.ec340dc94431b6b97167faa19657f751.jpg
 ┃ ┃ ┣ 📜nohelmet-7-_jpg.rf.e9b63f935ba5e871ac2352e1c3ad12cf.jpg
 ┃ ┃ ┣ 📜nohelmet-7-_png.rf.7f450085d4b2bd49e53d4b165891b273.jpg
 ┃ ┃ ┣ 📜nohelmet-70-_jpg.rf.1dd5a375e0542474b41400080e84d53d.jpg
 ┃ ┃ ┣ 📜nohelmet-71-_jpg.rf.b1f601f09106371bbe7af96c5c0bc9c6.jpg
 ┃ ┃ ┣ 📜nohelmet-73-_jpg.rf.bd0fbcccef5ad38463386baf060efdbd.jpg
 ┃ ┃ ┣ 📜nohelmet-74-_jpg.rf.a358a38e56a04965f85352577d67c4c8.jpg
 ┃ ┃ ┣ 📜nohelmet-76-_jpg.rf.90616cc956006acc71211338be195cf4.jpg
 ┃ ┃ ┣ 📜nohelmet-8-_jpg.rf.d6ebee54b6329179bf7b4474b91207e3.jpg
 ┃ ┃ ┣ 📜nohelmet-8-_png.rf.56ae71ad1d846cce915e85eba7ffb328.jpg
 ┃ ┃ ┗ 📜nohelmet-9-_png.rf.56f24f5bffbfdfdb36df7554ecd4cb02.jpg
 ┃ ┗ 📂labels
 ┃ ┃ ┣ 📜160610_jpg.rf.cf9ee04e91810899d842ef5feed9b700.txt
 ┃ ┃ ┣ 📜160613_jpg.rf.ace7457f0edabb2f94213250858579eb.txt
 ┃ ┃ ┣ 📜BikesHelmets13_png.rf.44b2421206bd428af238fa10cfc772f7.txt
 ┃ ┃ ┣ 📜BikesHelmets158_png.rf.e6879a189a98504a7ffb11a503725826.txt
 ┃ ┃ ┣ 📜BikesHelmets167_png.rf.e2de52c0b6346d1029b2f547c0bc7f7b.txt
 ┃ ┃ ┣ 📜BikesHelmets209_png.rf.bf3175b9a9ca43563d7d4bbe9a09f120.txt
 ┃ ┃ ┣ 📜BikesHelmets282_png.rf.d22efa640f3e6559093004497c4caa71.txt
 ┃ ┃ ┣ 📜BikesHelmets288_png.rf.04822f17bb0437ca3728a3eb154c6f4f.txt
 ┃ ┃ ┣ 📜BikesHelmets380_png.rf.7719b999f8dd72ed75ea156f946c1425.txt
 ┃ ┃ ┣ 📜BikesHelmets396_png.rf.48b485293de26b1c708d7b1ec9e0ae08.txt
 ┃ ┃ ┣ 📜BikesHelmets42_png.rf.7db6a6611e8fabc7e3beecedded42a9b.txt
 ┃ ┃ ┣ 📜BikesHelmets434_png.rf.a4a10cbb3e49fa97b729293e27cd89a7.txt
 ┃ ┃ ┣ 📜BikesHelmets451_png.rf.ce48ff6aff697f98a734b0b8b1c49152.txt
 ┃ ┃ ┣ 📜BikesHelmets468_png.rf.8297f31b73a89ca64f38d03d0181a2d8.txt
 ┃ ┃ ┣ 📜BikesHelmets476_png.rf.f76ae5571276a493ce10c4aab0aeaec6.txt
 ┃ ┃ ┣ 📜BikesHelmets508_png.rf.1aea1a9ec6ac416f88a8a0bed5824ed3.txt
 ┃ ┃ ┣ 📜BikesHelmets53_png.rf.3b92d7f1ff7352165ba6358df95b1a99.txt
 ┃ ┃ ┣ 📜BikesHelmets549_png.rf.b978a261301ebb860b8ce11c2e3ff96b.txt
 ┃ ┃ ┣ 📜BikesHelmets580_png.rf.50a0727292eeee9ac2cb07a68738b874.txt
 ┃ ┃ ┣ 📜BikesHelmets619_png.rf.2ca88fe2c43cc7c4a07c2f39a2e62ed5.txt
 ┃ ┃ ┣ 📜BikesHelmets629_png.rf.0033ba35c431e0e697c873117b6ee1de.txt
 ┃ ┃ ┣ 📜BikesHelmets653_png.rf.1b87c0477ca16f3941c2f94b87ebdd29.txt
 ┃ ┃ ┣ 📜BikesHelmets721_png.rf.30810b07a4e959aa4afe5f02831b790b.txt
 ┃ ┃ ┣ 📜BikesHelmets90_png.rf.bf5fc264672ae7df292081516d54e6d9.txt
 ┃ ┃ ┣ 📜BikesHelmets92_png.rf.72a5228e7ebda6d64e8f3df00ae845d9.txt
 ┃ ┃ ┣ 📜Image-22-_jpg.rf.3950dd59f63b678be71d230f485812c8.txt
 ┃ ┃ ┣ 📜Image-23-_jpeg.rf.1460483cbaa9346c580fd25eb444ec0b.txt
 ┃ ┃ ┣ 📜Image-24-_jpeg.rf.70d0ae0bf08db38a07d388e1f12da8cb.txt
 ┃ ┃ ┣ 📜Image-24-_jpg.rf.86fd0c6716364978fd27b34b9e2f48b5.txt
 ┃ ┃ ┣ 📜Image-25-_jpeg.rf.7427dcf5202a9129e74e71cfc7d81588.txt
 ┃ ┃ ┣ 📜Image-26-_jpeg.rf.57cebecc105b699d8f537f4e3cfa017c.txt
 ┃ ┃ ┣ 📜Image-28-_jpeg.rf.42c1a872709de30b8ba46b955428d254.txt
 ┃ ┃ ┣ 📜Image-28-_jpg.rf.c6a31f629bf5ace23b83d5b4cad20818.txt
 ┃ ┃ ┣ 📜Image-3-_jpg.rf.30e0eaa423b29ed9d64ccf30a304bf35.txt
 ┃ ┃ ┣ 📜Image-3-_png.rf.bf2fc1568d009618eed1832b7b0a8e8b.txt
 ┃ ┃ ┣ 📜Image-30-_jpeg.rf.64db4de98e8aac3779911d265eafb2dd.txt
 ┃ ┃ ┣ 📜Image-31-_jpeg.rf.44cc8028dc0e22886bc4ac5fa1422df3.txt
 ┃ ┃ ┣ 📜Image-33-_jpeg.rf.16768fa6fa82f91a112369c7a01ee2ec.txt
 ┃ ┃ ┣ 📜Image-34-_jpeg.rf.e53cdfc91b4e7fb3ea6e981309107107.txt
 ┃ ┃ ┣ 📜Image-39-_jpeg.rf.9811ac17db0b82c447a355079ce67977.txt
 ┃ ┃ ┣ 📜Image-39-_jpg.rf.63ce16d386e22b4eb09830a5661cdc59.txt
 ┃ ┃ ┣ 📜Image-4-_jpeg.rf.3309204fa4e6282af53e79c67c0b50a6.txt
 ┃ ┃ ┣ 📜Image-4-_jpg.rf.d5a26aa7c03daae5aca8641b936a5ab5.txt
 ┃ ┃ ┣ 📜Image-42-_jpeg.rf.fb2f054d1cf35460819e33d6aa4510f9.txt
 ┃ ┃ ┣ 📜Image-43-_jpg.rf.0846568e9dd34d32f687e9a35d560e63.txt
 ┃ ┃ ┣ 📜Image-44-_jpeg.rf.35641e5218f749921ea59b8fc69740ce.txt
 ┃ ┃ ┣ 📜Image-46-_jpeg.rf.09eff07d88ebb622a7da6ad484cf5648.txt
 ┃ ┃ ┣ 📜Image-46-_jpg.rf.2f8f2dcf14ac897a0db0bc0c13a896ca.txt
 ┃ ┃ ┣ 📜Image-47-_jpeg.rf.693adfa22f8604d19a959e23a450dc63.txt
 ┃ ┃ ┣ 📜Image-5-_jpeg.rf.b61aa14d08f2793979266e4758914161.txt
 ┃ ┃ ┣ 📜Image-5-_jpg.rf.9bebf09d483c171633eedbc003140745.txt
 ┃ ┃ ┣ 📜Image-50-_jpeg.rf.ec83f338f66fb6e69eda8e835404aa58.txt
 ┃ ┃ ┣ 📜Image-50-_jpg.rf.597bd890ba054ea0e74273cf57660eaa.txt
 ┃ ┃ ┣ 📜Image-51-_jpeg.rf.1da770c5d99d945e0d9a0f4460ac8c38.txt
 ┃ ┃ ┣ 📜Image-51-_jpg.rf.43bdc8b7250612c05e5aef6a3360a289.txt
 ┃ ┃ ┣ 📜Image-52-_jpeg.rf.7db136f49a216ec7448512bd8be3f85d.txt
 ┃ ┃ ┣ 📜Image-53-_jpeg.rf.45fa43aa0d6400dd902b55b9a8ad1980.txt
 ┃ ┃ ┣ 📜Image-54-_jpg.rf.3ecca0b2ce8036eb4c35cf542eb51b0e.txt
 ┃ ┃ ┣ 📜Image-56-_jpg.rf.4672163b5385a3b1596f1b0459c46cd1.txt
 ┃ ┃ ┣ 📜Image-6-_jpg.rf.a55c109ffddc87038d96653295fc4548.txt
 ┃ ┃ ┣ 📜Image-67-_jpg.rf.997b17a7a88353a774c20f6c5117b1be.txt
 ┃ ┃ ┣ 📜Image-68-_jpg.rf.634d2db1584203d680cf0d56d7f14d5e.txt
 ┃ ┃ ┣ 📜Image-69-_jpeg.rf.f54b2e051d5f298f1c54cead8f8fcb3b.txt
 ┃ ┃ ┣ 📜Image-70-_jpeg.rf.b5e1582df147426fc7d2755497b9f69e.txt
 ┃ ┃ ┣ 📜Image-70-_jpg.rf.2c656240f4f61b6478780a7cd7f04c27.txt
 ┃ ┃ ┣ 📜Image-71-_jpeg.rf.26d2fc54f6255608d99af76c700e157c.txt
 ┃ ┃ ┣ 📜Image-71-_jpg.rf.68f6ff884ccdd03e241d9179d10c6fe3.txt
 ┃ ┃ ┣ 📜Image-75-_jpg.rf.6e0fe9e2d49ef119018648dffad33f7e.txt
 ┃ ┃ ┣ 📜Image-8-_jpeg.rf.c86f41fcf180e9139f5a3b1fb11e9502.txt
 ┃ ┃ ┣ 📜Image-8-_jpg.rf.3a3fc1af44ac924651b5c0689949633f.txt
 ┃ ┃ ┣ 📜Image-9-_jpeg.rf.2f2377133a3912d1692d3620c6dc9664.txt
 ┃ ┃ ┣ 📜Image-97-_jpg.rf.e6ab3c968335ff83217e6fda2ab2c3aa.txt
 ┃ ┃ ┣ 📜nohelmet-10-_jpg.rf.3a723ac459368743f8a0b52f606ad412.txt
 ┃ ┃ ┣ 📜nohelmet-11-_jpg.rf.41537c8a25c3282b89701a7e521e74e2.txt
 ┃ ┃ ┣ 📜nohelmet-12-_jpg.rf.5ef447ee7a06a714b93301e7f4cfa092.txt
 ┃ ┃ ┣ 📜nohelmet-12-_png.rf.dd5d9c21b7379c603e15a9c0c418549b.txt
 ┃ ┃ ┣ 📜nohelmet-14-_jpg.rf.0b62e37e25c9896ae08eb1f7bd897e53.txt
 ┃ ┃ ┣ 📜nohelmet-14-_png.rf.16edb4bc5d0e16e85c80fca0390ddb6b.txt
 ┃ ┃ ┣ 📜nohelmet-15-_jpg.rf.9ff2467c17712b7f2d6ee93f0e05684b.txt
 ┃ ┃ ┣ 📜nohelmet-15-_png.rf.09d42d04dc0e988d51ed0eb9140efd89.txt
 ┃ ┃ ┣ 📜nohelmet-16-_jpg.rf.6f64dc830197fe4c5a34618cc310ffe5.txt
 ┃ ┃ ┣ 📜nohelmet-16-_png.rf.3c82164a6f786c59d9b9c94cd3aa213d.txt
 ┃ ┃ ┣ 📜nohelmet-17-_jpg.rf.765341a733365aadda335913b3ed83e1.txt
 ┃ ┃ ┣ 📜nohelmet-17-_png.rf.65f35c64920748129ea4a1d54cd59c99.txt
 ┃ ┃ ┣ 📜nohelmet-18-_jpg.rf.ec3ec65681de1eff87395420ce8c28c6.txt
 ┃ ┃ ┣ 📜nohelmet-18-_png.rf.54b257eb62c985af87fb62ba22819167.txt
 ┃ ┃ ┣ 📜nohelmet-19-_jpg.rf.bd773640223bfbc432ee56a7d7d512ab.txt
 ┃ ┃ ┣ 📜nohelmet-2-_jpg.rf.9890ce593237f65dd2c8555b170575aa.txt
 ┃ ┃ ┣ 📜nohelmet-20-_jpg.rf.65eb6247943d0eb4e8df332fbfff561a.txt
 ┃ ┃ ┣ 📜nohelmet-20-_png.rf.2a7300e1e8f517836115262a7620c098.txt
 ┃ ┃ ┣ 📜nohelmet-21-_jpg.rf.a4185b59bbf04fbc0887b42ca945ceea.txt
 ┃ ┃ ┣ 📜nohelmet-21-_png.rf.d0fa33d0915b88b926eb09ae847fd387.txt
 ┃ ┃ ┣ 📜nohelmet-22-_jpg.rf.9fb61668ecc5cb7132062efc77249a37.txt
 ┃ ┃ ┣ 📜nohelmet-22-_png.rf.94a8ebbc5fcb2e55fe94a412a44c43f8.txt
 ┃ ┃ ┣ 📜nohelmet-23-_jpg.rf.999079860d064d3ed890a066703e3fc5.txt
 ┃ ┃ ┣ 📜nohelmet-23-_png.rf.27a616fe48e3ba3952f149086daf3557.txt
 ┃ ┃ ┣ 📜nohelmet-24-_jpg.rf.69df39b5653c5e3cbec43107c3b8f3f5.txt
 ┃ ┃ ┣ 📜nohelmet-27-_jpg.rf.00989843a1285a82018dfc4773668301.txt
 ┃ ┃ ┣ 📜nohelmet-29-_jpg.rf.ce0b90b0476f2d827e06fd34ca34e201.txt
 ┃ ┃ ┣ 📜nohelmet-3-_jpg.rf.f9214eb29646840300ceb59f50256bdb.txt
 ┃ ┃ ┣ 📜nohelmet-3-_png.rf.d4ba340fe7c4656878dc4d1e23698871.txt
 ┃ ┃ ┣ 📜nohelmet-32-_jpg.rf.b9f59c0e1e87a07362f3420b780d62d0.txt
 ┃ ┃ ┣ 📜nohelmet-33-_jpg.rf.95ff09f84f64e40c324877f8e17e1729.txt
 ┃ ┃ ┣ 📜nohelmet-34-_jpg.rf.cd54689dccc22b07bb7a89e3f63b2fb6.txt
 ┃ ┃ ┣ 📜nohelmet-35-_jpg.rf.72ba77af56bee5e3b856ba596ae0fe24.txt
 ┃ ┃ ┣ 📜nohelmet-36-_jpg.rf.6fd05e137305e715cab13ca933a20d7b.txt
 ┃ ┃ ┣ 📜nohelmet-37-_jpg.rf.94222a991d6ce0a17a6ded66263f5011.txt
 ┃ ┃ ┣ 📜nohelmet-38-_jpg.rf.0899b0cad8995377ad1749c891e56dce.txt
 ┃ ┃ ┣ 📜nohelmet-39-_jpg.rf.9a11e99c61bbc236fd7c5fa36eac74c2.txt
 ┃ ┃ ┣ 📜nohelmet-4-_jpg.rf.1e2311c18fcfb0018d5bbf78d295512b.txt
 ┃ ┃ ┣ 📜nohelmet-4-_png.rf.8d63e3f04ded26426881a88c5a7f58e3.txt
 ┃ ┃ ┣ 📜nohelmet-40-_jpg.rf.8fbaa10454cb4ad4f1523796d24de3b0.txt
 ┃ ┃ ┣ 📜nohelmet-41-_jpg.rf.b6fbf9a011cf755e06a16656b95ff322.txt
 ┃ ┃ ┣ 📜nohelmet-44-_jpg.rf.d76174e99799321a92db2fef5975be54.txt
 ┃ ┃ ┣ 📜nohelmet-46-_jpg.rf.f142ef85cb270231e27f1393ade25770.txt
 ┃ ┃ ┣ 📜nohelmet-47-_jpg.rf.7de1809902b953a566915f77e6253433.txt
 ┃ ┃ ┣ 📜nohelmet-48-_jpg.rf.97e882e71117ab79c03421ac7905f956.txt
 ┃ ┃ ┣ 📜nohelmet-49-_jpg.rf.b1ce16b64180031b0e7e9362a229634f.txt
 ┃ ┃ ┣ 📜nohelmet-5-_png.rf.2ebb10c1a365886e7fda8eb2646117c5.txt
 ┃ ┃ ┣ 📜nohelmet-50-_jpg.rf.1e92fc16ec7278da0621df89000d073f.txt
 ┃ ┃ ┣ 📜nohelmet-52-_jpg.rf.6a1cb9bcfd8c31e500fea014374a36c9.txt
 ┃ ┃ ┣ 📜nohelmet-53-_jpg.rf.1a467d900fe5523104e45550a07f0d81.txt
 ┃ ┃ ┣ 📜nohelmet-55-_jpg.rf.8998ffec8a8f4b9fcf49ce583ce803aa.txt
 ┃ ┃ ┣ 📜nohelmet-56-_jpg.rf.3e6901961565139709393d4661815093.txt
 ┃ ┃ ┣ 📜nohelmet-57-_jpg.rf.241e7f8293d39be96d1499fc2165329b.txt
 ┃ ┃ ┣ 📜nohelmet-58-_jpg.rf.5eff0c8d81b380b5bf426f30fe7ebdfb.txt
 ┃ ┃ ┣ 📜nohelmet-6-_jpg.rf.3526de88d11496f28f459986be3167f8.txt
 ┃ ┃ ┣ 📜nohelmet-6-_png.rf.7972e4037e28e690025595435decee86.txt
 ┃ ┃ ┣ 📜nohelmet-60-_jpg.rf.7b7b0c87628ffe503695d489d9a29c7a.txt
 ┃ ┃ ┣ 📜nohelmet-61-_jpg.rf.fa957673f8d5d3779bbf1888e53f6276.txt
 ┃ ┃ ┣ 📜nohelmet-62-_jpg.rf.f95fb517f822743ace2c2fad5c4898cc.txt
 ┃ ┃ ┣ 📜nohelmet-65-_jpg.rf.d0da42d692e8f12f09e96b9e58b60433.txt
 ┃ ┃ ┣ 📜nohelmet-66-_jpg.rf.6a0e667c76beb7b957d03d370029e3c5.txt
 ┃ ┃ ┣ 📜nohelmet-68-_jpg.rf.30cda3c26f643cc4027b74a1cf541648.txt
 ┃ ┃ ┣ 📜nohelmet-69-_jpg.rf.ec340dc94431b6b97167faa19657f751.txt
 ┃ ┃ ┣ 📜nohelmet-7-_jpg.rf.e9b63f935ba5e871ac2352e1c3ad12cf.txt
 ┃ ┃ ┣ 📜nohelmet-7-_png.rf.7f450085d4b2bd49e53d4b165891b273.txt
 ┃ ┃ ┣ 📜nohelmet-70-_jpg.rf.1dd5a375e0542474b41400080e84d53d.txt
 ┃ ┃ ┣ 📜nohelmet-71-_jpg.rf.b1f601f09106371bbe7af96c5c0bc9c6.txt
 ┃ ┃ ┣ 📜nohelmet-73-_jpg.rf.bd0fbcccef5ad38463386baf060efdbd.txt
 ┃ ┃ ┣ 📜nohelmet-74-_jpg.rf.a358a38e56a04965f85352577d67c4c8.txt
 ┃ ┃ ┣ 📜nohelmet-76-_jpg.rf.90616cc956006acc71211338be195cf4.txt
 ┃ ┃ ┣ 📜nohelmet-8-_jpg.rf.d6ebee54b6329179bf7b4474b91207e3.txt
 ┃ ┃ ┣ 📜nohelmet-8-_png.rf.56ae71ad1d846cce915e85eba7ffb328.txt
 ┃ ┃ ┗ 📜nohelmet-9-_png.rf.56f24f5bffbfdfdb36df7554ecd4cb02.txt
 ┣ 📂valid
 ┃ ┣ 📂images
 ┃ ┃ ┣ 📜BikesHelmets169_png.rf.ca5345c7b614e805fa8fb0bcda242014.jpg
 ┃ ┃ ┣ 📜BikesHelmets246_png.rf.abfacf0f4a700f2255006f9f60b948e5.jpg
 ┃ ┃ ┣ 📜BikesHelmets428_png.rf.c49f7cc76dc2350b513ee7c446deb22e.jpg
 ┃ ┃ ┣ 📜BikesHelmets435_png.rf.a28581b6d584d12824e0fb4c2f4146c8.jpg
 ┃ ┃ ┣ 📜BikesHelmets572_png.rf.c014b913184ad6db8f3794268e9c80dd.jpg
 ┃ ┃ ┣ 📜BikesHelmets612_png.rf.b242f6c3895dc251e3787c968bdc5ac3.jpg
 ┃ ┃ ┣ 📜BikesHelmets749_png.rf.36c246d7daf10856e9b22c7af15b6e14.jpg
 ┃ ┃ ┣ 📜Image-107-_jpg.rf.c98b3adf66db509e1531b94a58e052cb.jpg
 ┃ ┃ ┣ 📜Image-29-_jpeg.rf.6cfe30d54d07011e28b8070fa5a23e5f.jpg
 ┃ ┃ ┣ 📜Image-4-_png.rf.167724488aae86037306ec8c33fedee9.jpg
 ┃ ┃ ┣ 📜Image-41-_jpg.rf.93d2c220ef74620bce5894dbccb5fa71.jpg
 ┃ ┃ ┣ 📜Image-44-_jpg.rf.883cb2ff39a2b0844a9c967f8282d39d.jpg
 ┃ ┃ ┣ 📜Image-45-_jpeg.rf.a9b848822d65302e6822dccc9296b1e7.jpg
 ┃ ┃ ┣ 📜Image-45-_jpg.rf.20153b9bae1959eea31cc16c2209ab22.jpg
 ┃ ┃ ┣ 📜Image-49-_jpeg.rf.6a2f330e9aa2b311a1d6ac757a3f0b20.jpg
 ┃ ┃ ┣ 📜Image-49-_jpg.rf.fa09581373c995b04bd7f6fb47eff7d3.jpg
 ┃ ┃ ┣ 📜Image-55-_jpg.rf.8f9a1298e1458c45bbb11f283861d487.jpg
 ┃ ┃ ┣ 📜Image-67-_jpeg.rf.162b3cfd2b56107d21454f8daa7815fc.jpg
 ┃ ┃ ┣ 📜Image-69-_jpg.rf.4d716703e7efbc180a901b056d921070.jpg
 ┃ ┃ ┣ 📜Image-7-_jpg.rf.3a3eef795bc4fad1328a11aa6b60d545.jpg
 ┃ ┃ ┣ 📜Image-96-_jpg.rf.18ac406ddacd3cd5132a47bfcd0a94cd.jpg
 ┃ ┃ ┣ 📜Image-98-_jpg.rf.ee9a26609f88322c358af3110d897d5b.jpg
 ┃ ┃ ┣ 📜nohelmet-1-_png.rf.25cc5d8b5aa148526ba530065821c4db.jpg
 ┃ ┃ ┣ 📜nohelmet-10-_png.rf.23490fbc271f8300f724bc3f82184b7e.jpg
 ┃ ┃ ┣ 📜nohelmet-13-_jpg.rf.88f6b8bb1b34db11ecdfd7f36a8672c5.jpg
 ┃ ┃ ┣ 📜nohelmet-13-_png.rf.9acaa5a70bec4a8af86fbb7ca954d71a.jpg
 ┃ ┃ ┣ 📜nohelmet-2-_png.rf.8ad8e72ebf47f04e1297ff6db01e05ea.jpg
 ┃ ┃ ┣ 📜nohelmet-25-_jpg.rf.17d43d7bf12c0ed2fcb3283d5ccbe421.jpg
 ┃ ┃ ┣ 📜nohelmet-26-_jpg.rf.b227e27ebdfb9353644b8bdf69d9c279.jpg
 ┃ ┃ ┣ 📜nohelmet-30-_jpg.rf.7587abb3bef4443f4aa8ce1b10573f8b.jpg
 ┃ ┃ ┣ 📜nohelmet-31-_jpg.rf.f2acb590b02b6a1038456777ef07b9fa.jpg
 ┃ ┃ ┣ 📜nohelmet-43-_jpg.rf.6077efc9cde9e76db9c81f27aaa291fc.jpg
 ┃ ┃ ┣ 📜nohelmet-45-_jpg.rf.d5811079a28f1038bad447201e7f0683.jpg
 ┃ ┃ ┣ 📜nohelmet-5-_jpg.rf.22c4eb81d9b5417c79070e86efb31a90.jpg
 ┃ ┃ ┣ 📜nohelmet-51-_jpg.rf.14809249309207ed7ff447eec718d27e.jpg
 ┃ ┃ ┣ 📜nohelmet-54-_jpg.rf.413c6101d164d02995da559dc17ab791.jpg
 ┃ ┃ ┣ 📜nohelmet-63-_jpg.rf.6b5eeadc5c32324ed91dbf49a0a3bee7.jpg
 ┃ ┃ ┣ 📜nohelmet-64-_jpg.rf.7184c1fa77da7c6f43b81c37e6deb694.jpg
 ┃ ┃ ┣ 📜nohelmet-67-_jpg.rf.666f32975ee39021b032c2f5e46acbcb.jpg
 ┃ ┃ ┣ 📜nohelmet-72-_jpg.rf.3148b2b72c10891bdf5117e60a502f95.jpg
 ┃ ┃ ┣ 📜nohelmet-75-_jpg.rf.fa29c979e17490ef8f59efbc623f99ec.jpg
 ┃ ┃ ┗ 📜nohelmet-9-_jpg.rf.2c58fd2d2af0beb351d48470415b57bb.jpg
 ┃ ┗ 📂labels
 ┃ ┃ ┣ 📜BikesHelmets169_png.rf.ca5345c7b614e805fa8fb0bcda242014.txt
 ┃ ┃ ┣ 📜BikesHelmets246_png.rf.abfacf0f4a700f2255006f9f60b948e5.txt
 ┃ ┃ ┣ 📜BikesHelmets428_png.rf.c49f7cc76dc2350b513ee7c446deb22e.txt
 ┃ ┃ ┣ 📜BikesHelmets435_png.rf.a28581b6d584d12824e0fb4c2f4146c8.txt
 ┃ ┃ ┣ 📜BikesHelmets572_png.rf.c014b913184ad6db8f3794268e9c80dd.txt
 ┃ ┃ ┣ 📜BikesHelmets612_png.rf.b242f6c3895dc251e3787c968bdc5ac3.txt
 ┃ ┃ ┣ 📜BikesHelmets749_png.rf.36c246d7daf10856e9b22c7af15b6e14.txt
 ┃ ┃ ┣ 📜Image-107-_jpg.rf.c98b3adf66db509e1531b94a58e052cb.txt
 ┃ ┃ ┣ 📜Image-29-_jpeg.rf.6cfe30d54d07011e28b8070fa5a23e5f.txt
 ┃ ┃ ┣ 📜Image-4-_png.rf.167724488aae86037306ec8c33fedee9.txt
 ┃ ┃ ┣ 📜Image-41-_jpg.rf.93d2c220ef74620bce5894dbccb5fa71.txt
 ┃ ┃ ┣ 📜Image-44-_jpg.rf.883cb2ff39a2b0844a9c967f8282d39d.txt
 ┃ ┃ ┣ 📜Image-45-_jpeg.rf.a9b848822d65302e6822dccc9296b1e7.txt
 ┃ ┃ ┣ 📜Image-45-_jpg.rf.20153b9bae1959eea31cc16c2209ab22.txt
 ┃ ┃ ┣ 📜Image-49-_jpeg.rf.6a2f330e9aa2b311a1d6ac757a3f0b20.txt
 ┃ ┃ ┣ 📜Image-49-_jpg.rf.fa09581373c995b04bd7f6fb47eff7d3.txt
 ┃ ┃ ┣ 📜Image-55-_jpg.rf.8f9a1298e1458c45bbb11f283861d487.txt
 ┃ ┃ ┣ 📜Image-67-_jpeg.rf.162b3cfd2b56107d21454f8daa7815fc.txt
 ┃ ┃ ┣ 📜Image-69-_jpg.rf.4d716703e7efbc180a901b056d921070.txt
 ┃ ┃ ┣ 📜Image-7-_jpg.rf.3a3eef795bc4fad1328a11aa6b60d545.txt
 ┃ ┃ ┣ 📜Image-96-_jpg.rf.18ac406ddacd3cd5132a47bfcd0a94cd.txt
 ┃ ┃ ┣ 📜Image-98-_jpg.rf.ee9a26609f88322c358af3110d897d5b.txt
 ┃ ┃ ┣ 📜nohelmet-1-_png.rf.25cc5d8b5aa148526ba530065821c4db.txt
 ┃ ┃ ┣ 📜nohelmet-10-_png.rf.23490fbc271f8300f724bc3f82184b7e.txt
 ┃ ┃ ┣ 📜nohelmet-13-_jpg.rf.88f6b8bb1b34db11ecdfd7f36a8672c5.txt
 ┃ ┃ ┣ 📜nohelmet-13-_png.rf.9acaa5a70bec4a8af86fbb7ca954d71a.txt
 ┃ ┃ ┣ 📜nohelmet-2-_png.rf.8ad8e72ebf47f04e1297ff6db01e05ea.txt
 ┃ ┃ ┣ 📜nohelmet-25-_jpg.rf.17d43d7bf12c0ed2fcb3283d5ccbe421.txt
 ┃ ┃ ┣ 📜nohelmet-26-_jpg.rf.b227e27ebdfb9353644b8bdf69d9c279.txt
 ┃ ┃ ┣ 📜nohelmet-30-_jpg.rf.7587abb3bef4443f4aa8ce1b10573f8b.txt
 ┃ ┃ ┣ 📜nohelmet-31-_jpg.rf.f2acb590b02b6a1038456777ef07b9fa.txt
 ┃ ┃ ┣ 📜nohelmet-43-_jpg.rf.6077efc9cde9e76db9c81f27aaa291fc.txt
 ┃ ┃ ┣ 📜nohelmet-45-_jpg.rf.d5811079a28f1038bad447201e7f0683.txt
 ┃ ┃ ┣ 📜nohelmet-5-_jpg.rf.22c4eb81d9b5417c79070e86efb31a90.txt
 ┃ ┃ ┣ 📜nohelmet-51-_jpg.rf.14809249309207ed7ff447eec718d27e.txt
 ┃ ┃ ┣ 📜nohelmet-54-_jpg.rf.413c6101d164d02995da559dc17ab791.txt
 ┃ ┃ ┣ 📜nohelmet-63-_jpg.rf.6b5eeadc5c32324ed91dbf49a0a3bee7.txt
 ┃ ┃ ┣ 📜nohelmet-64-_jpg.rf.7184c1fa77da7c6f43b81c37e6deb694.txt
 ┃ ┃ ┣ 📜nohelmet-67-_jpg.rf.666f32975ee39021b032c2f5e46acbcb.txt
 ┃ ┃ ┣ 📜nohelmet-72-_jpg.rf.3148b2b72c10891bdf5117e60a502f95.txt
 ┃ ┃ ┣ 📜nohelmet-75-_jpg.rf.fa29c979e17490ef8f59efbc623f99ec.txt
 ┃ ┃ ┗ 📜nohelmet-9-_jpg.rf.2c58fd2d2af0beb351d48470415b57bb.txt
 ┣ 📜data.yaml
 ┣ 📜README.dataset.txt
 ┗ 📜README.roboflow.txt
```

