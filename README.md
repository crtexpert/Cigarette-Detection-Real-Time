# Cigarette-Detection-Real-Time
YoloV5 Python


Bu doküman görüntü işleme tekniği kullanılarak videolarda ve gerçek zamanlı görüntülerde sigara objesinin tespit etmek ve ihtiyaç halinde sansürlemek için yapılmış araştırma hakkındadır. Türkiye’de ve dünyada farklı yasal düzenlemelerle medya kuruluşları için, düzeni sağlamak ve sakıncalı görsellerin paylaşımını engellemek gibi amaçlarla kısıtlamalar getirilmiştir. Sigara birçok ülkede sakıncalı ve medya yapımlarında gösterilmemesi gereken objeler arasında yer alıyor. İçeriklerinde bu tip yasaklı objelere yer veren kuruluşlara maddi para cezası ve yayın yasağı gibi kuruluşa hem parasal açıdan hem itibar açısından kötü etkileri bulunan cezalar veriliyor. Bu sebeple sigarayı tespit etmesi ve sansürlemesi için görüntü işleme tabanlı bir uygulama geliştirdim. Canlı yayınlarda müdahalesi zor ve insan gözünden kaçabilecek bu sorunu yapay zeka tabanlı bir yazılımla çözmek kurumlar için stres azaltıcı ve iş yükünü hafifleten bir seçenektir. Projem yapay zeka tabanlı, Pytorch kütüphanesini ve Yolo algoritmasıyla eğitilmiş makine öğrenmesi modellerini kullanarak çalışıyor. Farklı sigara markalarının farklı renklerde filtre ve kağıt yapısına sahip sigaralarını tespit edip sansür uyguluyor. Projeyi çok büyük bir veri seti üzerinden tasarlamak yerine, daha az veriyle iyi sonuçlar veren teknikler kullanarak tasarladım. Beyaz filtreli modellerde ayırt edici bir özellik bulunmadığı için kalem, fermuar gibi gündelik nesnelerle çok defa karışabiliyordu bu sebeple modelin devir sayısını fazla tutarak bu hataları en aza indirgemeyi amaçladım. Makine öğrenimi (MÖ) algoritmalarının verimliliğinin ve sınıflandırma performansının iyileştirilmesi açısından mevcut verilerden yüksek temsile sahip özniteliklerin çıkarılmasına özel önem verilmektedir [1]. Bu projeyi offline olarak çalışan masaüstü tabanlı bir proje olarak tasarladım. Pytorch kütüphanesi sayesinde modeli Python projeme yükleyebildim ve görüntüyü işlemek için OpenCV kütüphanesinden faydalandım. Program girdi olarak verilen görseli kare kare  işleme alıyor ve önceden eğitilmiş modelin içeriğine göre eşleşme bulması durumunda objenin x, y ekseni değerlerini ve yükseklik, genişlik değerlerini bir diziyle döndürüyor. Tespit edilen konuma bir çizim işlemi ve Gaussian Blur uygulanıyor.



BÖLÜM 2.	LİTERATÜR TARAMASI


Gelişen bilgisayar bilimleri ile birlikte temeli daha eskiye dayansa da büyük ilerlemelerin son dönemde olduğu bilgisayar görmesi kavramı hayatımıza dahil oldu. Bilgisayar görmesi fotoğraflar, videolar, metinler ve sesler gibi dijital verilerin bilgisayar tarafından anlaşılmasını ve işlenmesini geliştirmeyi amaçlayan bir bilim dalıdır. Yakın tarihte bilgisayar teknolojisinin ve donanımsal teknolojinin gelişmesiyle birlikte artık işlemler CPU ve GPU gücüyle çok daha kolay yapılabiliyor. Güçlü bilgisayarlar sayesinde eskiden metinler gibi daha düşük boyutta ve işlenmesi kolay veriler üzerinden gelişen kavram, artık yüksek çözünürlüklü resimler ve videolar veri seti olarak kullanılabiliyor ve işlenebiliyor. MÖ alanında fotoğraflardan objelerin spesifik yönleri bulunarak bir model oluşturulur ve bu sayede model kullanılarak tanıtılan nesnelerin tespiti yapılır.

Son dönemde görseller aracılığıyla nesnelerin sınıflandırılması işi neredeyse tamamen bilgisayarlara bırakıldı. Bilgisayarlar nesneleri anlamak ve yorumlamak açısından insan görüşünden daha iyi olmasa da hızlı işlem gücü ve insani hataları yapmaması sebebiyle bu alanda tercih edilmektedir.

Bilgisayarlar güçlü YZ algoritmaları kullanarak nesneleri tanıyabilir. Zaman geçtikçe veri setlerinin büyüklüğü bilgisayarları zorlamaya başladı. Görsellerin çözünürlüğünün artması dolayısıyla piksel sayılarının artması, görselleri işlerken yüksek işlem gücüne ihtiyaç duyulmasına neden oldu ve bu problemi çözmek için bilgisayarlarda GPU kullanımı yaygınlaştı.

ESA bu alanda nesneleri sınıflandırma ve hatırlama konuları için sık sık kullanılır. DÖ algoritmaları pikselleri matris yapılarına yerleştirerek sınıflandırır. Fotoğrafın bütün piksellerini matris yapısında yerleştirdikten sonra ilişkisel bir bağlantı arar ve çok yüksek işlem gücü gerektiren çok katmanlı matris işlemleri gerçekleştirir.
2.1. Dijital Görüntü İşleme

Dijital görüntü işleme 2 boyutlu resmin bilgisayar tarafından dijital bir forma çevrilmesi ve bu sayede görüntüden bir anlam çıkartılmasıdır. Görüntüyü dijital formata çevirmek için yazılımsal, donanımsal bir çok işlemden geçirilir ve bilgisayar hafızasında matris içerisinde ikili bitler olarak depolanır. 1920'lerde deniz altı kablosuyla alınan bir gazete fotoğrafının sayısallaştırılmasıyla görüntü işleme kavramı doğdu [2].

Görüntü işleme resimleri dijitale çevirme ve üzerinde bir takım işlemler gerçekleştirme sürecidir. Resmi işleyip yeni bir görsel oluşturmak veya resimden bir anlam çıkartma amaçlarıyla kullanılır. Dijital görüntü işleme, dijital bilgisayarlar kullanılarak resimlerin işlenmesinden oluşur. Kullanımları son yıllarda katlanarak artmaktadır. Uygulamaları, uzaktan algılama yoluyla geçen ilaçlardan ve jeolojik işlemeden eğlenceye kadar çeşitlilik gösterir [3]. Görseli görüntü işleme için kullanmadan önce bir dizi ön işlemden geçmesi gereklidir. Daha iyi sonuçlar almak için görselden gren de denilen parazitlerin ayıklanması gereklidir. Bu işlem birden fazla teknikle gerçekleştirilebilir. Görsele filtre uygulayarak sonucunda oluşacak görselden bilgisayarın bir anlam çıkarması kolaylaşır. Bu işlemlerin yapılmasının temelinde piksel sayısını azaltma gayesi yatar. Gereksiz pikseller görselden çıkarıldıktan sonra görüntü işleme işlemi daha kolay ve daha az işlem gücü kullanarak tamamlanabilir.

Görüntü işlemede gerekli prosedürler [4] .

1)	Edinme: dijital formda olan görüntüyü alma.
1.1)	Ölçekleme
1.2)	Renk çevirimi
2)	Görüntü Geliştirme: görseldeki yararlı veya gizli detayları çıkartma.
3)	Görüntü Restorasyonu: görüntüdeki bozulmaya bağlı olarak gerekli olabilir
4)	Renkli Görüntü İşleme
5)	Çoklu Çözünürlük İşleme: görseli değişik çözünürlüklerde sunmak.
6)	Görsel Sıkıştırma: düşük depolama sorunları için çözünürlükle ilgilenmek.
7)	Morfolojik İşleme
8)	Segmentasyon Prosedürü: resmi parçalara ayırmayı içerir.
9)	Temsil ve Açıklama: segmentasyon aşamasından çıktı çıkarır.
10)	Nesne Algılama ve Tanıma: Önceden belirlenmiş bir nesneye etiketleme işlemi.

2.1.1.	Görsel etiketleme

Bir resmin çekildiği her koşul için istisnai bir algoritma yapamayız, bu yüzden bir resim elde ettiğimizde onu genel bir algoritmanın çözmesini sağlayacak bir yapıya dönüştürürüz [5]. Görseller bilgisayarda matrislerde yüksek miktarlarda sayılarla depolandığı için bilgisayarın görselde ne olduğunu anlaması imkansızdır bu sebeple ortaya görsel etiketleme yöntemi çıkmıştır. 

![image](https://user-images.githubusercontent.com/55183922/218323016-7d5189ce-f791-4f8a-880f-020e6bc5d017.png)


Şekil 2.1. Makesense aracı kullanılarak etiketlenen görseller

Etiketleme işlemi YZ algoritmasının görsel içeriğini anlayabilmesi ve sonuçlar üretebilmesi için gereklidir.

2.2. Yapay Zeka

Endüstriyel alanda, sağlık alanında, sosyal yazılım uygulamaları gibi bir çok yazılımda, ileride insanların işlevlerini elinden alabileceği düşünülen MÖ, ESA konularıyla yakından ilişkili kavramdır. Algoritmik makine öğrenmesinde ve otonom karar alma konusunda yapılan keşiflerle birlikte, geliştirmede yeni yollar açılıyor [6].

YZ insan öğrenmesine çok benzer şekilde çalışır hatta taklit eder. Geçmiş deneyimlerden oluşan sonuçları yorumlar ve bunları değerlendirerek öğrenme sürecini taklit eder.

YZ teknolojisi günümüzde birçok alanda kullanılmaktadır. Ses tanıma sistemleri, otonom sürüş sistemleri, sesli asistanlar ve örneği çoğaltılabilecek bir çok alanda kullanılmaktadır. Bazı insanlar YZ entegreli sesli asistanlarla sohbet ediyor ve karşısında bir insan varmış gibi derdini anlatabiliyor. Bu durum yapay zeka uygulamalarının hayatımıza ne kadar çabuk dahil edildiğinin bir kanıtı olarak düşünülebilir. Bu yazılımlar insanların moral durumlarını analiz edebilir hatta geçmiş deneyimlerine dayanarak kişisel bazda özel bir iletişim bağı oluşturabilir. Bu nedenle, yapay zeka ve sınıflandırma tabanlı sistemler ile karmaşık prosedürler, yeniden tespit sonuçları üretmek için önerilmiştir [7]. YZ uygulamaları bu gibi bağları ve öğrenme işlevinin devamı için ESA ‘nı kullanır.

2.2.1. Makine öğrenmesi

Makine öğrenmesi önceden varolan bilgi üzerinde çıkarımlar yapıp gelecek hakkında tahminlerde bulunmak için veya bir durum hakkında istatistik oluşturmak için kullanılabilen, matematiği ve istatistikleri kullanarak sonuçlar ve modeler üreten bir dizi algoritma olarak tanımlanabilir.

Makine öğrenmesi kısaca cep telefonu, bilgisayar, arabalar gibi cihazların yazılımsal bir öğrenme yeteneği kazanmasıdır. Yeni nesil bütün akıllı cihazlarda artık somut bir yazılımdansa veriyi topladıktan sonra işleyen ve görevini duruma uygun şekilde yerine getiren makine öğrenmesi destekli yazılımlar kullanılıyor. Kullanılan algoritmalar sürekli olarak yeni veri üzerinde çıkarımlarda bulunup kendini ona göre optimize ederek sorunu çözmeyi öğreniyor.

Makine öğrenmesiyle birlikte şuana kadar yüz tanıma sistemleri, spam e-mail filtreleme, sağlıkta hastalık tespiti, hava tahminleri, otonom sürüş yazılımları, ses tanıma sistemleri, sahte belge tespiti gibi bir çok önemli konuda uygulamalar yapılmıştır.


Tablo 2.1. MÖ farklı türleri [8].
Makine Öğrenmesi Tipleri	Fonksiyonu	Örnek
Denetimli Öğrenme	Veri setinin etiketi ve çıktısı vardır, tahmin yapmak için veriden çıkarım yapar.	Yapay sinir ağları, Bayesian ağı
Denetimsiz Öğrenme	Veri setinin etiketi yoktur, ilişkileri tespit eder.	Temel bileşenler analizi, hiyerarşik kümeleme
Yarı-Denetimli Öğrenme	Denetimli ve denetimsiz öğrenmenin karışımıdır.	Resim ve ses tanıma
Yeniden Uygulamalı Öğrenme	İşlemleri gerçekleştirmek için ödül fonksiyonundan yararlanır.	Tıbbi görüntüleme


2.2.2. Sinir ağları

Sinir ağları teknolojinin insanlardan esinlendiği bir diğer kavramdır. İnsan beyninin çalışma prensibine benzer, nöron hücrelerinin birbirleriyle iletişimini dijital alanda modelleyen teknolojidir. Birbirine bağlı düğümlerden oluşur, ağın özniteliği topolojisine bakılarak söylenebilir. Kayıp fonksiyonunu en aza indirmek için, çıktı ile gerçek çıktı arasındaki farkı tespit etmek için eğitim verileri kullanılır. Ortaya çıkan bir hata varsa nöronların ağırlıklarının güncellenmesi işlemi yapılır. Güncelleme işlemi geri yayılım algoritması kullanılarak yapılır [9].

İlk defa 1943 yılında Warren McCulloh ve Walter Pitts tarafından ortaya atılmıştır. 1940’lı yılların ilerleyen dönemlerinde Donald Olding Hebb, Hebbian Öğrenmesi olarak da bilinen, beyindeki sinir ağlarının iletişimini taklit eden bir hipotez oluşturmuştur.

1959 yılına gelindiğinde, Marchian Hoff ve Bernard Widrow örüntüyü tanıyabilen modeler oluşturdular ve telefon hattındaki bir sonraki biti tahmin edebildiler.

Dünyaca başarılı olduğunun kabullenilişi 1995 yılında borsa tahminlerinde ve otonom araçlar için kullanılmasıyla oldu.

Günümüze kadar bir çok MÖ ve DÖ ‘de kullanılan birçok yapay sinir ağı geliştirildi. Bunlardan en yaygın olanları; İleri Beslemeli Yapay Sinir Ağları (İBYSA), Radyal Tabanlı Yapay Sinir Ağları (RTYSA), Yinelemeli Yapay Sinir Ağları (YYSA) ve Evrişimli Yapay Sinir Ağları (ESA) olmak üzere 4 e ayrılır.

2.2.2.1. İBYSA

Sinir ağları arasında en ilkel olanı İBYSA’dır. YYSA’nın tam zıttı olarak çalışır. Düğümler bir döngü içerisinde değildir. Tek yönde ilerler ve asla geri dönmez, veri birkaç saklı düğümden geçtikten sonra yoluna devam eder.
 
 ![image](https://user-images.githubusercontent.com/55183922/218323145-ba7b89dd-7278-42cb-99f9-d3e8bc25fbab.png)

Şekil 2.2. İleri Beslemeli Yapay Sinir Ağı çalışma prensibi [10]


Bu modelde bir dizi girdi katmana girer ve model değeriyle (W) çarpılır. Daha sonra bu değerler toplanır. Eğer bu toplam eşik değerinin üzerinde çıkarsa (eşik değeri değişken olarak ayarlanabilir, ayarlanmadığı durumda 0 olarak kabul edilir) 1 çıktısı alınır, eğer eşik değerinin altındaysa -1 çıktısı alınır.

İBYSA genellikle denetimli öğrenme metodunu kullanır. Etiketleme işlemi yapılmış veri setleri üzerinden tahmin yapar. Genellikle fiyat tahmini gibi daha basit uygulamalarda tercih edilir.

2.2.2.2. RTYSA

Radyal Tabanlı Ağlar sinir ağlarının bir örneğidir. Diğer ağlardan farklı olarak tek bir kullanım durumu için özelleştirilmiştir. RTYSA, radyal fonksiyonu esas alan ve doğrusal olmayan sınıflandırma içeren durumlar için özelleşmiştir.
![image](https://user-images.githubusercontent.com/55183922/218323156-c5f4a4e9-6289-42f4-bf3b-926140e1c766.png)

Şekil 2.3. Radyal tabanlı yapay sinir ağı çalışma prensibi [11].

RTYSA sınıflandırılacak giriş değeri, radyal tabanlı fonksiyon nöronları, ve çıktı düğümleri olmak üzere 3 ana parçadan oluşmaktadır. Her bir nöron eğitilen modelle ilgili prototip vektörü adı verilen bilgileri taşır. 1 çıktısı girdi değeriyle prototip değerin eşleştiğini gösterir. Girdi değeriyle prototip değer arasındaki fark açıldıkça çıktı 0 değerine hızla düşer. Bu değer aktivasyon değeri olarak da bilinir.
 ![image](https://user-images.githubusercontent.com/55183922/218323164-8b0fc538-9a5d-45b7-ba29-3ff2bad88540.png)

Şekil 2.4. İki kategorili veri setine dayanan RTYSA örneği [11].
2.2.2.3. YYSA

Yinelemeli Yapay Sinir Ağları veriyi depolamak için döngüleri kullanan bir yapay sinir ağıdır. Önceki deneyimlerinden çıkarımlar yaparak gelecek hakkında çıkarımda bulunur. YYSA bu yapısından dolayı ses tanıma, çeviri gibi uygulamalarda tercih edilir. Girdi olarak aldığı bir cümle için, pozitif yada negative duygu çıkarımı yapabilir veya girdi olarak bir görüntü alıp çıktı olarak bir cümle verebilir.

 ![image](https://user-images.githubusercontent.com/55183922/218323173-bf27607e-6bc1-4cbf-9920-6a7e3dd0ffa1.png)

Şekil 2.5. Yinelemeli Yapay Sinir Ağı çalışma prensibi [10].


2.2.2.4. ESA

Evrişimli Yapay Sinir Ağı bilgisayar görmesi uygulamalarında en sık kullanılan yapay sinir ağıdır. Resimler gibi yapılandırılmış veri dizilerini işlemek için üretilmiştir. ESA’nın resimlerdeki çizgiler, daireler, yüzler gibi şablonları bulmak konusundaki üstünlüğü bilgisayar görmesi programlarında tercih edilmesine sebep oluyor. ESA, İBYSA’nın bir türüdür. Yineleme Katmanı adında özel bir katmana sahiptir. Yineleme katmanları üst üste geldikçe daha karmaşık şekilleri tanıyabilir örneğin 4,5 yineleme katmanlı bir ESA el yazımı harfleri tanıyabilirken, 30 yineleme katmanlı bir ESA insan suratını tanıyabilir.


 ![image](https://user-images.githubusercontent.com/55183922/218323181-695f0a39-1756-488b-b17e-f45e657845ee.png)

Şekil 2.6. Evrişimli sinir ağları katman yapısı [12].



2.3. Yolo 

Yolo (You Look Only Once) yüksek performanslı bir ESA modelidir. Yolo görüntülerdeki nesneleri gerçek zamanlı olarak tespit etmek için kullanılır. Yolo görüntüyü ızgara şeklinde parçalara böler ve her bir parça objeyi kendi içerisinde tespit eder.
 
 ![image](https://user-images.githubusercontent.com/55183922/218323192-0905efc5-abda-4753-9308-5454a427e371.png)

Şekil 2.7. Yolo ızgara sistemi [13].


Yolo objenin tespiti için tek bir ESA kullanır ve her bir ızgara bölmesi için tek tek  tahmin yapar. Bütün hücreler eş zamanlı olarak tahmin yapar. Geleneksel yöntemlere göre çok daha hızlıdır ve tahminlerin doğruluğunu arttırmak için bütün resmi inceler.

Yolo ağı iki tamamen bağlı katmandan ve 24 ESA’dan oluşur. GoogLeNet tarafından kullanılan başlangıç modülleri yerine yalnızca 1 x 1 indirgeme katmanlarını ve ardından 3 x 3 konvolüsyon katmanlarını kullanırlar [14].

 ![image](https://user-images.githubusercontent.com/55183922/218323200-5b107373-f663-400d-8aa4-a0023dad436b.png)

Şekil 2.8. Yolo mimarisi.


Yolo’nun en yeni versiyonu YoloV5’dir. YoloV5’in yapısı 3 parçada değerlendirilebilir. Omurga (CSPDarknet), boyun (PANet), ve kafa (Yolo katmanı). İlk olarak veriler omurga aracılığıyla gönderilir daha sonra boyun kısmı bu verileri bir araya getirir ve son olarak kafa kısmında tespit edilen objeler görüntülenir [15].


 ![image](https://user-images.githubusercontent.com/55183922/218323206-e2831df2-76c7-4a0b-adfc-ee8359f0774e.png)

Şekil 2.9. YoloV5 mimarisi.
BÖLÜM 3.	MALZEMELER VE YÖNTEMLER

DÖ teknikleri ile oluşturulmuş sistemler, modellerini topladıkları verileri kendileri işleyerek oluşturabilir fakat MÖ ile oluşturulmuş sistemler veri ihtiyacını insanlar olmadan karşılayamaz. Verilerin etiketlenmesi ve bilgisayar için anlamlı bir duruma getirilmesi gereklidir. Güvenlik kameraları gibi sürekli veri toplayabilen cihazlar için DÖ tabanlı sistemler tercih edilir. Tezime konu olan proje için MÖ tabanlı algoritmalar ve ESA tabanlı Yolo tekniğini kullandım.

DÖ algoritmaları çok daha büyük bir veri havuzuna ihtiyaç duyar ve bu sebeple yüksek işlem gücü için GPU kullanır. Tezime konu olan projede uygulamayı orta ölçekli cihazlarda da çalışabilecek, GPU gücüne ihtiyaç duymadan daha uzun sürede de olsa CPU gücü ile çalışabilecek boyutlarda veri setlerini kullanarak tasarladım. Daha az veri programın emin olma eşiğini aşağı çekse de performans yönlü tercihlerde daha az veriyle de başarılı sonuçlar elde edilebilir. 

3.1. Veri Eldesi

Sigara objesi markaların farklı boyutlarda, renklerde ve şekillerde ürettiği bir ürün olduğu ve şekli gündelik hayatta her ortamda bulunabilecek nesnelere benzeyebildiği için veri seti geniş tutulması gerekiyor. Giriş kısmında da bahsettiğim gibi daha küçük boyutlu ama etkili bir model için veri setindeki fotoğrafları ücretsiz stok görsel sağlayan bir servis ile elde ettim. Farklı kameralarla çekilmiş, farklı ışık koşullarında, farklı markaların sigaralarının bulunduğu görselleri topladım.

3.2. Etiketleme İşlemi

MÖ sistemleri veri hakkındaki bilgiyi dışardan alır. Matris içerisinde tuttuğu verinin hangi bölümünde tanıtılmak istenen nesnenin bulunduğu bilgisini etiketleme yöntemi sayesinde alır ve işler. Etiketleme işlemi sonucunda txt dosyalarında sınıf numarası ve etiketlenen kısmın x ekseni, y ekseni, genişlik ve yükseklik değerleri tutulur. Etiketleme işlemi için Python içinde Labelmg gibi kendi kütüphaneleri bulunsa da projemde internet üzerinde bulunan online etiketleme servisi olan makesense.ai ‘ı tercih ettim. 
 
 ![image](https://user-images.githubusercontent.com/55183922/218323219-a2421914-a8f0-4102-b860-ebf648207b4c.png)

Şekil 3.1. Sigara veri seti için etiket correlogramı.

3.3. Model Oluşturma

Modeli oluştururken cihazımda güçlü bir GPU bulunmadığı için ücretsiz GPU ve RAM kullanımı sunan ve arayüz olarak da kolaylık sunan Google Colaboraty servisini kullandım. Veri havuzuna eğitim için 180 görsel, doğrulama işlemi için 20 görsel ekledim. Model eğitim aşamasında GPU ‘nun kaldırdığı maksimum değerin biraz altında olan 16 batch (parti) boyutunu kullandım. Devir sayısı olarak modeli hafif tutmak amaçlı 100 devir sayısı ile başladım. Fakat modelin eminliği için yetersiz bir sayı olduğu için devir sayısını arttırarak ilerledim ve 0.995 maP50 eminlik değerine 300 devirli eğitim turunda ulaştım.

 ![image](https://user-images.githubusercontent.com/55183922/218323227-02c62507-ec5e-447e-9095-da919292183d.png)

Şekil 3.2 Model eğitim sonucu mAP05 değerleri.

 ![image](https://user-images.githubusercontent.com/55183922/218323236-18fc3080-a466-4deb-bd0a-03ed325f0a9a.png)

Şekil 3.3 Model eğitim sonucu mAP05 eminlik değerleri.

Yolo’nun birçok önceden eğitilmiş kontrol noktaları bulunuyor. Bunlar YoloV5n, YoloV5s, YoloV5m, YoloV5l, YoloV5x gibi ihtiyaca göre seçilebilecek farklı boyutlardaki kontrol noktaları. YoloV5x daha büyük işlem gücüne ihtiyaç duyuyor fakat YoloV5s ‘dan daha emin tahminlerde bulunabiliyor. YoloV5s daha çok mobil uygulamalar ve düşük sistem gereksinimleri için oluşturulmuş bir kontrol noktası olduğu için eğitim işleminde YoloV5s’yi tercih ettim.

 ![image](https://user-images.githubusercontent.com/55183922/218323237-de71a10b-c0ee-4db2-b6c8-e6a2c7d493df.png)

Şekil 3.4. Eğitim süresi boyunca nesnellik kaybı değerleri.


3.4. Arayüz Tasarımı

Hazırladığım modelin kolay kullanımı için Python’un içerisinde bulunan Tkinter kütüphanesinden faydalandım. Tkinter kütüphanesi sayesinde ilkel bir menü tasarımı yaptım ve menüyü gerçek zamanlı görüntü işleme ve video işleme olarak 2 farklı sahneye ayırdım.


 ![image](https://user-images.githubusercontent.com/55183922/218323243-093809f5-7d1d-42cd-a0e9-09abffefb840.png)

Şekil 3.5. Arayüz tasarımı.


 ![image](https://user-images.githubusercontent.com/55183922/218323247-b5d8d8dc-a1aa-472c-aff9-53e9dd5c7554.png)

Şekil 3.6. Arayüz video tespit bölümü.







BÖLÜM 4.	DENEY



Tezimde daha önce deneyimlemediğim görüntü işleme ve bilgisayar görmesi alanında Yolo algoritmasının kullanarak sigara tespit ve sansürleme uygulaması geliştirdim. Kullandığım veri seti bilgisayar görmesi sistemleri için çok dar bir veri kümesi içerse de yolo algoritması sayesinde iyi sonuçlar elde ettim.


4.1. 	Ağ Testleri

Bir ESA geliştirdikten sonra ağın doğru çalıştığının tespit edilmesi , farklı test verilerinin doğurduğu farklı sonuçlar göz önüne alınarak ve antrenman aşamasında kullanılan veri setinin ne kadar genel bir durum serisine uygun olduğu tespit edilmesi gerekir.

Modeli hazırladıktan sonra veri setinin yeterli olmamasından kaynaklı şekil 4.1. ‘de göründüğü gibi hatalı sonuçlar elde edilebiliyordu.

 ![image](https://user-images.githubusercontent.com/55183922/218323258-c9416de3-0f5b-4897-8b34-5ba75b5dae93.png)

Şekil 4.1. Hatalı sonuç örneği.


Bu sonuçları geliştirmek ve hata payını en aza indirgemek için modeli 100 , 200 , 300 gibi farklı sayı adetlerinde, modeli geliştirmede kullanılanın veriler haricinde veri setlerinde içerisinde sigaranın  bulunmadığı fotoğrafların da olduğu bir teste soktum. Böylelikle şekil 4.2. ‘de olduğu gibi daha isabetli şekilde video, fotoğraf ve eş zamanlı şekilde sigara tespitini ve sansürlenmesini gerçekleştirebildim

![image](https://user-images.githubusercontent.com/55183922/218323263-9187c56f-ee65-4daf-8b4a-71f67c16860c.png)

Şekil 4.2. Stabil sigara tespiti örneği.

4.2. 	Test Sonuçları

Her test aşamasında mAP0.5 (mean average precision) değeri olarak bilinen R-CNN , SSD gibi obje tespit algoritmalarında çokça kullanılan 0 ile 1 değerleri arasında doğruluk seviyesini ifade eden metrikleri elde ettim. Daha önceki bölümde de bahsettiğim gibi doğrulama aşamasında içerisinde sigaranın bulunmadığı fotoğrafların da bulunduğu farklı veri setlerinin, her geçen testte daha kararlı mAP0.5 değerleri ortaya koyduğunu aşağıda bulunan 3 tabloyla görebiliriz.


 ![image](https://user-images.githubusercontent.com/55183922/218323274-292d2856-ff3c-4c87-87b9-1c0876c3b859.png)

Şekil 4.3. İlk test için mAP0.5 grafiği.
![image](https://user-images.githubusercontent.com/55183922/218323279-39f7b101-dcf0-4bf6-ab70-8eccae615f52.png)

 
Şekil 4.4. İkinci test için mAP0.5 grafiği.

![image](https://user-images.githubusercontent.com/55183922/218323282-fcb9fbf1-d10f-4694-a050-97e030aef05f.png)

 
Şekil 4.5. Son test için mAP0.5 grafiği.


Tablolardan da anlaşılacağı üzere her test aşamasında model daha kararlı ve güvenilir hale gelmiş oldu.

Tablolarda bulunan mAP0.5 değerlerinin yüksekliği testlerde elde edilen kayıp fonksiyonlarına bağlıdır. Son test aşamasında modelin box_loss ( kutu kaybı ) değeri 0.0025 ve obj_loss ( obje kaybı ) değeri 0.04 sınırlarının altına indi. Tezin önceki kısımlarında da anlattığım gibi yolo algoritması ızgara sistemiyle çalışır ve ızgaraların her biri içerisinde obje özel olarak aranır. Kayıp fonksiyonları bu ızgaralarda obje tespit edilmesi durumunda ceza işlemi uygular ve kayıp puanını düşürür. Son teste ait box_loss tablosu şekil 4.6. ve obj_loss tablosu şekil 4.7. ‘da gösterildiği gibidir.

 ![image](https://user-images.githubusercontent.com/55183922/218323290-2ee3f4fe-c93a-4009-ba6a-0c154847fd92.png)

Şekil 4.6. Kutu kaybı grafiği.


 ![image](https://user-images.githubusercontent.com/55183922/218323294-7bec5eea-eaa6-4906-8bdf-edd5984b70e7.png)

Şekil 4.7. Obje kaybı grafiği.


Test sonuçlarında elde edilen bir diğer değer de Precision-Recall eğrisidir. Recall değeri arttıkça precision değeri azalır. Çünkü her bir obje tespitinde modelin hassasiyeti azalır ve hatırlama değeri artmış olur. Bir sonraki tabloda hassasiyet değerinin hatırlama değerine , hassasiyet değerinin eminlik değerine oranı ve F1 eğrisinin grafiği görülebilir.

 ![image](https://user-images.githubusercontent.com/55183922/218323298-d12c8146-470e-4bcc-8f9c-27a52c7116e9.png)

Şekil 4.8. F1/Confidence grafiği.


 ![image](https://user-images.githubusercontent.com/55183922/218323304-1320ec82-3831-45db-9dc0-8da3164b1533.png)

Şekil 4.9. Recall/Confidence grafiği.
BÖLÜM 5. SONUÇ

Sigara insan sağlığına düzenli kullanımda veya pasif içicilik yoluyla kalıcı hasarlar veren bir tütün mamülü olmasının yanında kullanımı ve satışı birkaç ülke dışında serbesttir. Sigara markaları ürünlerini çeşitli renklerde ve boyutlarda , şirin ve kullanıma özendirecek paketlerde sunarak insanları bu ürünü kullanmaya çekmektedir. Tütün mamüllerini yasaklamak bir seçenek olmasa da bu tarz zararlı maddelerin yaygınlığını azaltmak ve reşit olmayan yaştaki bireylerin sigara ve diğer tütün mamüllerine özenmesine engel olmak için denetleyici kurumlar tarafından yayın ve medya kuruluşlarına sansür uygulaması getirilmiştir. İçinde Türkiyenin de bulunduğu bazı ülkelerde kapalı alanlarda sigara kullanımı da yasaklanmıştır fakat bu kısıtlar insan kontrolünde olduğu için yetkili kişiler tarafından esnetilebilmektedir. Bu yüzden ileride sadece medya kuruluşlarında değil aynı zamanda kısıtlamaların olduğu yerlerde kameralar tarafından bu yasaklar otomasyon haline getirilebilir. 

Çalışmamda Yolo eklentisini kullanarak geliştirdiğim model ve OpenCv kütüphanesiyle görüntüleri anlık olarak işleyen, Python diliyle yazılan masaüstü uygulaması sayesinde video, resim ve gerçek zamanlı olmak üzere 3 farklı görsel türünde 97% eminlik seviyesinde sigara tespiti yapılabiliyor. Uygulama için geliştirdiğim ESA, geliştirmelerle birlikte sigaranın tespit edilmesi gereken durumlarda bulunan mobese kameraları gibi bir çok teknolojik yapıya entegre edilebilir ve gerçek zamanlı olarak sigara tespiti yapılabilir ve bu ağ ile birlikte sigaranın pasif içiciliği ve özendirilmesi engellenebilir. Bu sayede sigara kullanmayan insanların istemeden sigara dumanına maruz kalması ve çocukların erken yaşta sigarayla tanışması engellenebilir.








KAYNAKLAR



[1]		Bengio, Y., A.Courville, and P.Vincent. 2013. Representation learning: A review and new perspectives. IEEE Trans. Pattern Anal. Mach. Intelligence, 35, 8., 1798–1828.

[2]		Perihanoglu, G. 2015. Feature extraction from images by using digital image processing techniques, Master’s thesis. Istanbul Technical Univ.

[3]		Eduardo A.B. da Silva, G. V. M. (2005). The Electrical Engineering Handbook (W.-K. CHEN Ed.).

[4]		Gonzales, R. C., & Woods, R. E. (2008). Digital image processing (third ed.): Prentice hall New Jersey.

[5]		Sonka, M., Hlavac, V., & Boyle, R. (1993). Image pre-processing. In Image Processing, Analysis and Machine Vision (pp. 56-111): Springer.

[6]		Dwivedi, Y. K., Hughes, L., Ismagilova, E., Aarts, G., Coombs, C., Crick, T., Duan, Y., Dwivedi, R., Edwards, J., Eirug, A., Galanos, V., Ilavarasan, P. V., Janssen, M., Jones, P., Kar, A. K., Kizgin, H., Kronemann, B., Lal, B., Lucini, B., . . . Williams, M. D. 2021. Artificial intelligence (ai): Multidisciplinary perspectives on emerging challenges, opportunities, and agenda for research, practice and policy. International Journal of Information Management, 57, 101994. https://doi.org/https: //doi.org/10.1016/j.ijinfomgt.2019.08.002.

[7]		Waleed, M., Um, T.-W., Khan, A., & Khan, U. J. R. S. (2020). Automatic Detection System of Olive Trees Using Improved K-Means Algorithm. 12(5), 760.

[8]		Seetharam, K., D, B., PD, F., and PP, S. 2020. Role of artificial intelligence in cardiovascular imaging: State of the art review. front. Front. Cardiovasc. Med. https: //doi.org/https://doi.org/10.3389/fcvm.2020.618849 
		
[9]		LeCun, Y., Boser, B., Denker, J., D.Henderson, R.E.Howard, W.E.Hubbard, and Jackel, L. 1990. Handwritten digit recognition with a backpropagation network. In Advances in neural information processing systems, 396–404.
		
[10]		Venelin Valkov https://medium.com/@curiousily/making-a-predictive-keyboard-using-recurrent-neural-networks-tensorflow-for-hackers-part-v-3f238d824218.

		
[11]		Chris McCormick https://mccormickml.com/2013/08/15/radial-basis-function-network-rbfn-tutorial/
		
[12]		Thomas Wood, DeepAi “https://deepai.org/machine-learning-glossary-and-terms/convolutional-neural-network” Erişim Tarihi : 23.12.2022

[13]		Alexey Bochkovskiy, Chien-Yao Wang, Hong-Yuan Mark Liao https://arxiv.org/abs/2004.10934v1
		
[14]		Abrol, S., and Mahajan, R. 2015. Artificial neural network implementation on fpga chip. Int J Comput Sci Inf Technol Res, 3, 11–18.
		
[15]		Xu, R., Lin, H., Lu, K., Cao, L., and Liu, Y. 2021. A forest fire detection system based on ensemble learning. Forests, 12, 217.
