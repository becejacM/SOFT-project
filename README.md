# SOFT-project
Sign language recognition

**Autor:**

- SW10/2014 Milana Becejac

**Definicija problema:**

Potrebno je detektovati šaku sa video kamere i prepoznati koje je to slovo alfabeta ili je razmak, 
zatim ispisati reci na ekranu ili u fajl.

**Motivacija problema:**

Rešenje problema može biti korisno za sporazumevanje gluvo-nemih osoba, koje koriste sign-language
i osoba koje ovaj jezik ne znaju i samim tim pruži ovim osobama više poslovnih mogucnosti i omoguci im lakši život.

**Skup podataka:**

Skup podataka ce biti ručno napravljen pomoću kamere laptopa. Biće napravljen skup fotografija šake, 
koje predstavljaju po jedno slovo alfabeta. Za svako slovo 300-400 fotografija. Takodje, biće napravljen skup fotografija koje predstavljaju razmak i koje ne predstavljaju ništa. Fotografije ce biti učitane i prosedjene neuronskoj mrezi za treniranje(70%), validaciju(10%) i testiranje(20%).
Dataset se moze preuzeti preko ovog linka: https://drive.google.com/open?id=1Vv29P9f61c1esPFYHs447wZf-ywud8qC

**Metodologija:**

- Za implementaciju koristiće se Python i njegova biblioteka OpenCV

- Za treniranje koristice se Convolution Neural Network

- Radi će se prepoznavanje boje koze sa videa priemenom maske sa skin color range-om,
 kako bi se detektovala šaka i izdvojila od pozadine. Zatim ce se vršiti predikcija koje je slovo alfabeta
pokazano i formiranje slova i reci.

**Metod evaluacije:**

- Validacija rešenja pomocu testnog skupa slika koje predstavljaju jedan znak, tj jedno slovo azbuke. 
Koristice se mera tacnosti(Accurancy) za ocenjivanje kvalitetć modela. 

- Takodje biće odradjeni i manualani testovi gde će se meriti procenat ispravno prepoznatih slova i procenat uspešno prepoznatih reci.

- Celi vdei na kojima je prikazan demo aplikacije se mogu preuzeti preko ovog linka: https://drive.google.com/open?id=1ho7GpufvRvaOlzsYBKfNDeL943K32TmJ
(gifovi su ograničeni na 20 sec)

**Primena projektnog resenja:**

Projektno rešenje se može primeniti u sistemima za komunikaciju nemih osoba i onih koje to nisu.

**Demo**

- hello_world: https://j.gifs.com/E9XyjK.gif
- i_love_you: https://j.gifs.com/RoKXRV.gif
- demo 1: https://j.gifs.com/vox5GV.gif
- demo 2: https://j.gifs.com/L81DQp.gif
- demo 3: https://j.gifs.com/jqgVRy.gif
