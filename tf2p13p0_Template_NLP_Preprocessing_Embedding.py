#numpy==1.24.3
#pandas==2.0.3
#Pillow==10.0.0
#scipy==1.10.1
#tensorflow==2.13.0
#tensorflow-datasets==4.9.2
#matplotlib=3.7.2
#scikit-learn=1.3.0
#nltk=3.8.1



# load data from internet and preprocess it




############### Import libraries
import requests
from bs4 import BeautifulSoup
import re
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


import nltk



### get data from internet
#url = "https://en.wikipedia.org/wiki/Machine_learning"
#response = requests.get(url)
#soup = BeautifulSoup(response.content, 'html.parser')
#passage = " ".join([p.text for p in soup.find_all('p')])


#### Read data from text file
filename = "C:/Users/PRAU4KBR/Documents/PyCharm_Projects/TensorFlow_Certificate/datasets/Raw_Text_MachineLearningWikipedia.txt"
passage = ""
with open(filename, 'r') as file:
    data = file.read().replace('\n', ' ')
    passage = passage+data

## get stopwords from nltk
stopwords = nltk.corpus.stopwords.words('english')

# to lowercase
passage = passage.lower()

# remove special characters
def remove_special_characters(text):
    text = re.sub(r'http\S+', ' ', text )
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\bhttps?://[a-zA-Z0-9-]+(?:\.[a-zA-Z0-9-]+)+\b', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    text = re.sub(r'\d', ' ', text)  # Corrected line
    text= re.sub(r'[\u4e00-\u9fff]+', ' ', text)
    return text

passage = remove_special_characters((passage))



######## Tokenization
# word level tokenization
tokenizer = tf.keras.preprocessing.text.Tokenizer()

tokenizer.fit_on_texts([passage])

print("Showing tokenizers")
print(tokenizer.word_index)
print("\n\n\n")

print("Showing Sequence of words")
sequence = tokenizer.texts_to_sequences([passage])
print(sequence)


########################################### Tokenization again
sentences = ["I love reading books",
             "The cat sat on the mat",
             "It's a beautiful day outside",
             "Have you done your homework?",
             "Machine Learning is a very interesting subject that enables you build amazing solutions beyond your imagination."]

# remove special characters and use lowercase
for i, s in enumerate(sentences):
    s = remove_special_characters(s)
    s = s.lower()
    sentences[i] = s

# word embedding
tokenizer_word = tf.keras.preprocessing.text.Tokenizer(oov_token="<OOV>") # define out of vocabulary words token
tokenizer_char = tf.keras.preprocessing.text.Tokenizer(char_level = True) #characters

tokenizer_word.fit_on_texts(sentences)
print("Showing tokenizers - Words")
print(tokenizer_word.word_index)
print("\n\n\n")

print("Showing Sequence of words")
sequence_word = tokenizer_word.texts_to_sequences(sentences)
print(sequence_word)


### padded sequence of words
max_length = 10
sequence_word_padded = tf.keras.preprocessing.sequence.pad_sequences(sequence_word, padding='post', maxlen=max_length, truncating='post')
print("\n\n\n Padded Sequence of Words")
print(sequence_word_padded)

##### use trained tokenizer to tokanize new sentences
test_sentence = ["I love playing chess"]

# remove special characters and use lowercase
for i, s in enumerate(test_sentence):
    s = remove_special_characters(s)
    s = s.lower()
    test_sentence[i] = s

test_sequence = tokenizer_word.texts_to_sequences(test_sentence)
test_sequence_padded = tf.keras.preprocessing.sequence.pad_sequences(test_sequence, padding='post', maxlen=max_length, truncating='post')


###################################### Word Embeddings####################################
# get sample text for training

sample_text = ["I'll text you my address.",
"He pulled out his phone to text Damian.",
"Jule's text message brought him back to the unpleasant task ahead of him.",
"She shook her head and sat down, sending a text to Darian.",
"He typed a quick text then shut off the phone.",
"This morning she asked me the meaning of carpenter, and the question furnished the text for the day's lesson.",
"His phone had a text waiting.",
"The text message was from Jule.",
"Another text popped up, worsening his mood.",
"Text me if you need anything else.",
"An unread text message blinked on the screen. 4got 2 tell you.",
"Sofia pulled over to the side of the road to await the text and load the address into the car's GPS.",
"Xander received a text a minute later.",
"A text made her phone ding as she finished eating.",
"They won't let me in, Ashley's text was accompanied by half a dozen frowny faces.",
"Xander reached for his phone, satisfied to see a text from Jule.",
"Feel free to call, text or visit.",
"She pulled it free to see a text from Dusty.",
"They'll be freaked out, but I'll text them and tell them to go with you.",
"Careful study of the text will not support this view.",
"In the last-mentioned work he seeks to prove that the St Petersburg Codex, for so many years accepted as the genuine text of the Babylonian school, is in reality a Palestinian text carefully altered so as to render it conformable to the Babylonian recension.",
"As Hebrew became less familiar to the people, a system of translating the text of the Law into the Aramaic vernacular verse by verse, was adopted in the synagogue.",
"Both Talmuds are arranged according to the six orders of the Mishnah, but the discussion of the Mishnic text often wanders off into widely different topics.",
"Its object was to fix the biblical text unalterably.",
"It is generally divided into the Great and the Small Masorah, forming together an apparatus criticus which grew up gradually in the course of centuries and now accompanies the text in most MSS.",
"Some system of the kind was necessary to guard against corruptions of copyists, while the care bestowed upon it no doubt reacted so as to enhance the sanctity ascribed to the text.",
"Their long lists of the occurrences of words and forms fixed with accuracy the present (Masoretic) text, which they had produced, and were invaluable to subsequent lexicographers, while their system of vowel-points and accents not only gives us the pronunciation and manner of reading traditional about the 7th century A.D., but frequently serves also the purpose of an explanatory commentary.",
"The Hebrew text was edited with a Latin translation by Breithaupt (Gotha, 1707).",
"Nitzsch, however, held that this was a copyist's gloss, harmonizing with the received Boetius legend, which had been transferred to the text, and did not consider that it outweighed the opposing internal evidence from De Cons.",
"The play is, however, founded on Bacon's Life, of which the text is used by Ford with admirable discretion, and on Thomas Gainsford's True and Wonderful History of Perkin Warbeck (1618).",
"Mitchell reprinted the 1567 volume (expurgated) for the Scottish Text Society.",
"The text of the Complaynt was first edited by Leyden in 1801.",
"The true character of Urim (as expressing  aye ) and Thummim (as expressing  nay ) is shown by the reconstructed text of 1 Sam.",
"The text of the notice of the third Cadmus of Miletus in Suidas is unsatisfactory; and it is uncertain whether he is to be explained in the same way, or whether he was an historical personage, of whom all further record is lost.",
"The text was published in 20 vols.",
"In these illustrations, which gave an impluse to the production of enblems and were copied in the English version, there appears a humour quite absent from the text.",
"He particularly congratulated himself on having discovered the  philosophical argument  against transubstantiation,  that the text of Scripture which seems to inculcate the real presence is attested only by a single sense - our sight, while the real presence itself is disproved by three of our senses - the sight, the touch, and the taste.",
"With regard to the second of the above complaints, surprise will probably be felt that it was not extended to portions of the text as well as to the notes.",
"Before Lightfoot's time commentaries, especially on the epistles, had not infrequently consisted either of short homilies on particular portions of the text, or of endeavours to enforce foregone conclusions, or of attempts to decide with infinite industry and ingenuity between the interpretations of former commentators.",
"That the substance of the Physiologus was borrowed from commentaries on Scripture 4 is confirmed by many of the sections opening with a text, followed up by some such formula as but the Physiologus says.",
"When zoological records failed, Egypto-Hellenic ingenuity was never at a loss for a fanciful invention distilled from the text itself, but which to succeeding copyists appeared as part of the teaching of the original Physiologus.",
"So little was the collection considered as a literary work with a definite text that every one assumed a right to abridge or enlarge, to insert ideas of his own, or fresh scriptural quotations; nor were the scribes and translators by any means scrupulous about the names of natural objects, and even the passages from Holy Writ.",
"The Greek text of the Physiologus exists only in late MSS., and has to be corrected from the translations.",
"From the text which Philo uses, it is probable that the translation had been transmitted in writing; and his legend probably fixes the date of the commencement of the undertaking for the reign of Ptolemy Lagus.",
"This would easily allow clarification of what is 100-years-ago vague without really tampering with the text.",
"It even appears from a study of the Greek text that some copies of the books of Samuel incorporated narratives which other copies did not acknowledge.",
"According to the Hebrew text of I Sam.",
"The Gibeonites demanded the latter, and five sons of Merab (the text by a mistake reads Michal) and two sons of Saul's concubine were sacrificed.",
"There is no critical edition, and the only version available for the general reader is the modernized and abridged text published by Paulin Paris in vols.",
"This only begins with what Paulin Paris terms the Agravain section, all the part previous to Guenevere's rescue from Meleagant having been lost; but the text is an excellent one, agreeing closely with the Lenoire edition of 1533.",
"The text at his disposal, especially in the Queste section, must have been closely akin to that used by the Dutch translator and the compiler of Lenoire, 1533.",
"In other cases he tampers with the documents which he inserts (as, for instance, with the text of Magna Carta).",
"The original works of Rufinus are - (I) De Adulteratione Librorum Origenis - an appendix to his translation of the Apology of Pamphilus, and intended to show that many of the features in Origen's teaching which were then held to be objectionable arise from interpolations and falsifications of the genuine text; (2) De Benedictionibus XII Patriarcharum Libri II - an exposition of Gen.",
"A reprint of 1670 is only valuable because it contains P. de Fermat's notes; as far as the Greek text is concerned it is much inferior to the other.",
"Origen's textual studies on the Old Testament were undertaken partly in order to improve the manuscript tradition, and partly for apologetic reasons, to clear up the relation between the LXX and the original Hebrew text.",
"The results of more than twenty years' labour were set forth in his Hexapla and Tetrapla, in which he placed the Hebrew text side by side with the various Greek versions, examined their mutual relations in detail, and tried to find the basis for a more reliable text of the LXX.",
"Origen worked also at the text of the New Testament, although he produced no recension of his own.",
"It is, moreover, highly probable that he was the author of a radical pamphlet entitled La Philosophie au people frangais, published in 1788, the text of which is not known.",
"C. Thomas, accompanying the Latin text, with full biographical and bibliographical introductions (1888).",
"The work is almost wholly a compilation, and that not of the most discriminative kind, while a peculiar jealousy of Gesner is continuously displayed, though his statements are very constantly quoted - nearly always as those of  Ornithologus, his name appearing but few times in the text, and not at all in the list of authors cited.",
"Latham entered, so far as the limits of his work would allow, into the 1 They were drawn and engraved by Martinet, who himself began in 1787 a Histoire des oiseaux with small coloured plates which have some merit, but the text is worthless.",
"Heysham added to Hutchins's Cumberland a list of birds of that county, whilst in the same year began Thomas Lord's valueless Entire New System of Ornithology, the text of which was written or corrected by Dr Dupree, and in 1794 Donovan began a History of British Birds which was only finished in 1819 - the earlier portion being reissued about the same time.",
"The first volume of this, containing the land-birds, appeared in 1797 6 - the text being, it is understood, by Beilby - the second, containing the water-birds, in 1804.",
"Of the text it may be said that it is respectable, but no more.",
"Temminck, whose father's aid to Le Vaillant has already been noticed, brought out at Paris a Histoire naturelle des pigeons illustrated by Madame Knip, who had drawn the plates for Desmarest's volume.3 Since we have begun by considering these large illustrated works in which the text is made subservient to the coloured plates, it may be convenient to continue our notice of such others of similar character as it may be expedient to mention here, though thereby we shall be led somewhat far afield.",
"It does not seem to have been the author's original intention to publish any letterpress to this enormous work, but to let the plates tell their own story, though finally, with the assistance, as is now known, of William Macgillivray, a text, on the whole more than respectable, was produced in five large Ma egil- octavos under the title of Ornithological Biography, of liyr ay.",
"A Century of Birds from the Himalaya Mountains was followed by The Temminck subsequently reproduced, with many additions, the text of this volume in his Histoire naturelle des pigeons et des gallinacees, published at Amsterdam in 1813-1815, in 3 vols.",
"In this the large plates were reduced by means of the camera lucida, the text was revised, and the whole systematically arranged.",
"Nevertheless a scientific character was so adroitly assumed that scientific men - some of them even ornithologists - have thence been led to believe the text had a scientific value, and that of a high class.",
"It now behoves us to turn to general and particularly systematic works in which plates, if they exist at all, form but an accessory to the text.",
"The former of these has the entire text, but no plates; the latter reproduces the plates, but the text is in places much condensed, and excellent notes are added.",
"The procession was followed, inside the church, by a curious combination of ritual office and mystery play, the text of which, according to the Ordo processionis asinorum secundum Rothomagensem usum, is given in Du Cange.",
"Their opportunity came with the disaster which befell the Roman army under Valerian (q.v.) at Edessa, a disaster, says ' The full text, both Greek and Palmyrene, with an English translation, is given in NSI, pp. 313-340.",
"The text of Ibelin became a textus receptus - but it also became overlaid by glosses, for it was used as authoritative in the kingdom of Cyprus after the loss of the kingdom of Jerusalem, and it needed expounding.",
"At the same time, if our text is thus late, it must be remembered that its content gives us the earliest and purest exposition of French feudalism, and describes for us the organization of a kingdom, where all rights and duties were connected with the fief, and the monarch was only a suzerain of feudatories.",
"This text, however, is not a law, but rather an abstract of the special usages obtaining in those regions - what the Germans call a Weistum.",
"Linda whipped out a phone to text Lon.",
"Of the seven treatises contained in the Abhidhamma Pitaka five, and one-third of the sixth, had by 1910 been published by the Pali Text Society; and one, the Dhamma Sangani, had been translated by Mrs Rhys Davids.",
"Franke in two articles in the Journal of the Pali Text Society for 1903, and in his Geschichte and Kritik der einheimischen Pali Grammatik.",
"Two volumes only of these, out of about twenty still extant in MS., have been edited for the Pali Text Society.",
"Besides some occasional references in the text, only a few more of the general works dealing with the distribution of birds can here be mentioned.",
"They are important witnesses to the text of the New Testament, to the history of the canon, and to the history of interpretation.",
"Various scholars, while agreeing on the actual divisions of the text, differ on the question of priority.",
"He commented on all the Bible and on nearly all the Talmud, has been himself the text of several super-commentaries, and has exercised great influence on Christian exegesis.",
"They were used, like the assizes of the high court, in Cyprus; and, like the other assizes, they were made the subject of investigation in 1531, with the object of discovering a good text.",
"A change of the Hebrew text seems necessary; possibly we should read S1p $t', low is the voice, instead of 51p$ o'p', he rises up at the voice.",
"An edition of the Arabic text has been printed at Bulaq, (7 vols., 1867) and a part of the work has been translated by the late Baron McG.",
"The best maps are those in Die Karten von Attika, published with explanatory text by the German Archaeological Institute (Berlin, 1881).",
"The earlier text, of which five short fragments have come down to us, is known as the Pactus Alamannorum, and from the persistent recurrence of the expression et sic convenit was most probably drawn up by an official commission.",
"There is no doubt that the text dates back to the reign of Dagobert I., i.e.",
"The later text, known as the Lex Alamannorum, dates from a period when Alamannia was independent under national dukes, but recognized the theoretical suzerainty of the Frankish kings.",
"Germany comprised two other duchies, Saxony and Frisia, of each of which we possess a text of law.",
"This text is a collection of local customs arranged in the same order as the law of the Ripuarians.",
"The oldest mention of Robin Hood at present known occurs in the second edition - what is called the B text - of Piers the Plowman, the date of which is about 1377.",
"If Marca's criticism is too often undecided, both in the ancient epochs, where he supports the text by a certain amount of guesswork and in certain points where he touches on religion, yet he always gives the text correctly.",
"The text of Gildas founded on Gale's edition collated with two other MSS., with elaborate introductions, is included in the Monumenta historica Britannica, edited by Petrie and Sharpe (London, 1848).",
"Jameson (1897) of the text of the Ring (first published in the pocket edition of the full scores) is the most wonderful tour de force yet achieved in its line.",
"There has been much controversy both as to the authenticity of some of the sermons in this edition and as to the text in general.",
"In his text Eratosthenes ignored the popular division of the world into Europe, Asia and Libya, and substituted for it a northern and southern division, divided by the parallel of Rhodes, each of which he subdivided into sphragides or plinthia - seals or plinths.",
"Charles's Text, § 12.",
"The existence of these Christian elements in the text misled nearly every scholar for the past four hundred years into believing that the book itself was a Christian apocryph.",
"We have already shown that St Paul twice quoted from the Greek text of the Testaments.",
"For the text of scripture he uses both the Latin versions, the Itala and the Vulgate, often comparing them together.",
"His text, however, is so confused, both from obscurity of style and from corruptions in the MSS., that there is much difference of opinion as to the meaning of many words and phrases employed in his narrative, and their application in particular points of detail.",
"Besides, in case of the entire roll not being filled with the text, the unused and inferior sheets at the end could be better spared, and so might be cut off.",
"For the catacombs of Alexandria, Neroutsos Bey, L'Ancienne Alexandrie, may be consulted in addition to De Rossi's article mentioned in the text.",
"But it will be noticed that the second half of the definition in the text - from the general premisses of all reasoning - is left unexpressed.",
"An important event not to be passed over without mention is the grant on the 10th of March 1870 of the firman instituting the Bulgarian exarchate, thus severing the Bulgarian Church from Text in Holland, p. 212.",
"This seeming casual connexion, to some extent, confirms the historic connexion suggested by the text, that the Jews at the Exodus had to use bread prepared in haste; but not even Hebrew tradition attempts to explain why the abstention should last for seven days.",
"This is a highly ingenious hypothesis to explain the discrepancies of the text, but is, after all, nothing but hypothesis.",
"Either of these will supply the names of works upon Clement's biblical text, his use of Stoic writers, his quotations from heathen writers, and his relation to heathen philosophy.",
"Though now cultivated in India, and almost wild in some parts of the northwest, and, as we have seen, probably also in Afghanistan, it has no Sanskrit name; it is not mentioned in the Hebrew text of the Scriptures, nor in the earliest Greek times.",
"The text of this letter occurs in a number of MSS.",
"An English translation by the side of the Welsh text of the so-called triads of Dyvnwal Moel Mud is given by Owen, in the The Ancient Laws of Wales.",
"The Latin text is much shorter than the Welsh, but we do not know whether this abridgment was made on purpose, or whether the translation is an imitation of an earlier text.",
"Ulm is remarkable in the history of German literature as the spot where the Meistersinger lingered longest, preserving without text and without notes the traditional lore of their craft.",
"The quatrains have been edited at Calcutta (1836) and Teheran (1857 and i862); text and French translation by 3.",
"Occasionally the word  invariants  includes covariants; when this is so it will be implied by the text.",
"An incisive introduction discusses the ecclesiastical tradition, modern criticism; the second, the first and the third Gospels; the evangelical tradition; the career and the teaching of Jesus; and the literary form, the tradition of the text and the previous commentaries.",
"The quotations from the Old Testament are made from the Massoretic text.",
"The text has been edited most completely by Bonnet, Acta Apostol.",
"According to this authority one-third of the text is now lost.",
"The best text is that of Lipsius, Acta Apostol.",
"The text is in the utmost confusion. It is unreadable.",
"The text of the Actus Vercellenses is edited by Lipsius, Ada Apostol.",
"The best critical edition of the Greek text will be found in Lipsius, Acta Apostolorum Apocrypha, 1891, pp. 279-283.",
"The text has been edited by Hilgenfeld in 1877, Gebhardt and Harnack in 1878, and Funk in 1887 and 1901.",
"The text of the passages has to be critically treated anew.",
"The last-named gives an elaborate history of interpretation from the Septuagint down to Calvin, and appends the Ethiopic text edited by Dillmann.",
"The text, however, is in a very corrupt state.",
"On the other hand, there are elements in the poem which show that it is not entirely the work of a poor crowder; and these (notably references to historical and literary authorities, and occasional reminiscences of the literary tricks of the Scots Chaucerian school) have inclined some to the view that the text, as we have it, is an edited version of the minstrel's rough song story.",
"In 1889 the Scottish Text Society completed their edition of the text, with prolegomena and notes by James Moir.",
"Adrien Augier resumed the work, giving Lebeuf's text, though correcting the numerous typographical errors of the original edition (5 vols., 1883), and added a sixth volume containing an analytical table of contents.",
"A single sentence in Porphyry's Isagoge or  introduc tion  to the Categories of Aristotle furnished the i o, s text of the discussion.",
"The Teubner text by Naber is based on this.",
"And here we have first to observe that in the Hebrew text the Psalter is divided into five books, each of which closes with a doxology.",
"The musical notes found in the titles of the psalms and occasionally also in the text (Selah, 1 Higgaion) are so obscure that it seems unnecessary to enter here upon the various conjectures that have been made about them.",
"The Hexaplar text of the LXX., as reduced by Origen into greater conformity with the Hebrew by the aid of subsequent Greek versions, was further the mother (d) of the Psalterium gallicanum - that is, of Jerome's second revision of the Psalter (385) by the aid of the Hexaplar text; this edition became current in Gaul and ultimately was taken into the Vulgate; (e) of the SyroHexaplar version (published by Bugati, 1820, and in facsimile from the famous Ambrosian MS. by Ceriani, Milan, 1874).",
"This important version was first published in a good text by Lagarde, Psalterium juxta hebraeos hieronymi (Leipzig, 1874).",
"While some works of patristic writers are still of value for text criticism and for the history of early exegetical tradition, the treatment of the Psalms by ancient and medieval Christian writers is as a whole such as to throw light on the ideas of the commentators and their times rather than on the sense of a text which most of them knew only through translations.",
"Beginning with the earliest versions of the Bible, which seem to date from the 2nd century A.D., the series comprises a great mass of translations from Greek originals - theological, philosophical, legendary, historical and scientific. In a fair number of cases the Syriac version has preserved to us the substance of a lost original text.",
"The text of the Gospels underlying it  represents the Greek text as read in Rome about A.D.",
"Its text  represents, where it differs from the Diatessaron, the Greek text as read in Antioch about A.D.",
"Rabbi - la's text of the Gospels  represents the Greek text as read in Antioch about A.D.",
"The history of the Peshitta rendering of the Acts and Epistles is less clear; apparently the earliest Syrian writers used a text somewhat different from that which afterwards became the standard.'",
"Rabbula, the powerful and energetic bishop of Edessa who withstood the beginnings of Nestorianism, and who gave currency to the Peshitta text of the four Gospels, abolishing the use of the Diatessaron, is dealt with in a separate article.",
"The entire text of the London MS. was published by Land in the third volume of his Anecdota syriaca; and there is now an English translation by Hamilton and Brooks (London, 1899), and a German one by Ahrens and Kruger (Leipzig, 1899).",
"The Syriac text is rendered from a Greek original of unknown age, which from its complete correspondence with the Key of Truth may be judged to have been a Paulician writing.",
"They explained away baptisms as words of the Holy Gospel, citing the text I am the living water.",
"Their canon included only the Gospel and Apostle, of which they respected the text, but distorted the meaning.",
"But the character being ideographic, the words which express them are dissimilar in the two languages, and official text is read in Chinese by a Chinese, in Annamese by an Annamese.",
"To him Damasus entrusted the revision of the Latin text of the Bible and other works of religious erudition.",
"There is no doubt that his work is chiefly a compilation; and Daremberg, with other scholars, has traced a large number of passages of the Latin text to the Greek originals from which they were translated.",
"Van Swieten's commentaries on the aphorisms of Boerhaave are thought more valuable than the original text.",
"The scientific editing of the text began with C. C. Lachmann (1852) whose work still holds the field.",
"In the second place, in direct disregard of a promise given to Frederick, a supplement to Akakia appeared, more offensive than the main text.",
"So you should indicate precisely, what parts of your site are restricted in use â€” because the 1911 text as such (whether on paper or in electronic form) is free, and anyone may use it for any purpose, without any conditions.",
"He contributed to draw up Louis's charter, and in his memoirs boasted of having furnished the text of the proclamation addressed by the king to the French people before his return to France; but it is known now that it was another text that was adopted.",
"From this text we learn that the Dynasty of Ur consisted of five kings and lasted for 117 years, and was.",
"Lehmann-Haupt .uggested an emendation of the text, reducing the number by a thousand years; 14 while Winckler has regarded the statement of Nabonidus as an uncritical exaggeration.",
"So, for -example, the word for  name  may be written by a sign MU, or it may be written out by two signs shu-mu, the one sign MU representing the  Sumerian  word for  name, which, however, in the case of a Babylonian or Assyrian text must be read as shumu - the Semitic equivalent of the Sumerian MU.",
"The third text consists of 99 chapters, and is divided into two groups, according as the MSS.",
"Opinions differ as to the true import of these glosses; some scholars hold that the Salic Law was originally written in the Frankish vernacular, and that these words are remnants of the ancient text, while others regard them as legal formulae such as would be used either by a plaintiff in introducing a suit, or by the judge to denote the exact composition to be pronounced.",
"Even the most ancient text, that in 65 chapters, contains passages which a comparison with the later texts shows to be interpolations.",
"Finally, we find capitularies of the kings immediately following Clovis being gradually incorporated in the text of the lawe.g.'the Pactum pro tenore pacis of Childebert I.",
"The text is filled with valuable information on the state of the family and property in the 6th century, and it is astonishing to find Montesquieu describing the Salic Law as the law of a people ignorant of landed property.",
"Geffcken, Lex Salica (Leipzig, 1898), the text in 65 chapters, with commentary paragraph by paragraph, and appendix of additamenta; and the edition undertaken by Mario Krammer for the Mon.",
"There is an edition of the text of the Ripuarian Law in Mon.",
"It was compiled by the itinerant Frankish officials known as the missi Dominici, and the text undoubtedly goes back to the time of Charlemagne, perhaps to the years 802 and 803, when the activity of the missi was at its height.",
"Now it is acknowledged by Christian and Jewish scholars alike to have been written in Hebrew in the 2nd century B.C. From Hebrew it was translated into Greek and from Greek into Armenian and Slavonic. The versions have come down in their entirety, and small portions of the Hebrew text have been recovered from later Jewish writings.",
"A Christian revision of it is probably preserved in the two dialects of Coptic. Of these the Akhmim text is the original of the Sahidic. These texts and their translations have been edited by Steindorff, Die Apokalypse des Elias, eine unbekannte Apokalypse and Bruchstiicke der Sophonias-Apokalypse (1899).",
"There are other Hebraisms in the text.",
"His sermons occasionally created some stir, and on one occasion Elizabeth interrupted his sermon, telling him to stick to his text and cease slighting the crucifix.",
"The numerous copies of Odoric's narrative (both of the original text and of the versions in French, Italian, &c.) that have come down to our time, chiefly from the 14th century, show how speedily and widely it acquired popularity.",
"He loved discussing the sense of single portions of the Bible with other scholars, and made many fine expositions of the text.",
"Somewhat curiously, but very naturally, Enoch the son of Cain is confused with the Enoch who was translated to heaven - an error which the author of the Old English Genesis avoids, though (according to the existing text) he confounds the names of Enoch and Enos.",
"That the author of the Heliand was, so to speak, another Ca dmon - an unlearned man who turned into poetry what was read to him from the sacred writings - is impossible, because in many passages the text of the sources is so closely followed that it is clear that the poet wrote with the Latin books before him.",
"It included the original text and the variations of it dating from the 12th, 13th and 14th centuries.",
"Though he sometimes glided lightly over difficulties, his work is of service in fixing the text of Tabari.",
"Sturz (1824-1836); text by I.",
"The same qualities appear in Walton's Considerator Considered (1659), a reply to the Considerations of John Owen, who thought that the accumulation of material for the revision of the received text tended to atheism.",
"In the middle of the pages is the Latin text of the Bible; in the margins are the glosses, consisting of a very full collection of patristic excerpts in illustration and explanation of the text.",
"The name of the ruler alluded to is not indeed introduced into the actual text, but Carolus Inlperator form the initial letters of the passage dealing with this subject.",
"In such cases of substitution the vowels of the word which is to be read are written in the Hebrew text with the consonants of the word which is not to be read.",
"In the printed text this document, entitled An Invective Against the Armenians, is dated 800 years after Constantine, but the author Isaac Catholicos almost certainly belonged to the earlier time.",
"A valuable edition of the De aquis (text and translation) has been published by C. Herschel (Boston, Mass., 1899).",
"Lord Palmerston soon saw that further resistance was useless; his Peelite colleagues stuck to their text, and, within three weeks after resuming office, Gladstone, Sir James Graham and Mr Sidney Herbert resigned.",
"The text shows a curious mingling of sources; the real primitive Perceval story, the Enfances, is omitted; he grows up in his father's house and goes to court at his wish.",
"This was at one time claimed as the original source of all the Perceval romances, but this theory cannot be maintained in face of the fact that the writer gives in one place what is practically a literal translation of Chretien's text in a passage which there is strong reason to believe was borrowed by Chretien from an earlier poem.",
"The Perceval was edited from the Mons text by Potvin (6 vols., 1866-1871); Syr Percyvelle of Galles, in The Thornton Romances, by Halliwell (1844) for the Camden Society.",
"For the general reader the most useful text is that of Bartsch in Deutsche Classiker des Mittelalters, as it includes notes and a glossary.",
"The Welsh text, with translation, has been edited by Canon Williams. A fine translation by Dr Sebastian Evans is published in The Temple Classics, under the title of The High History of the Holy Grail.",
"They marked him as one of the most able critics of Bentley's (in many cases) rash and tasteless conjectural alterations of the text.",
"He was thus enabled to go to; Italy to study the Vatican text of Plutarch, on the translation on whose Lives (1 559; 1 565) he had been some time engaged.",
"This piece was played after the fall of the Terror, but the fratricide of Timoleon became the text for insinuations to the effect that by his silence Joseph de Chenier had connived at the judicial murder of Andre, whom Joseph's enemies alluded to as Abel.",
"Its vigour and originality have had scanty justice done to them owing to the difficulty of the subject-matter and the style, and the corruptions which still disfigure its text.",
"A first edition of his Historia Britonum was in circulation by the year 1139, although the text which we possess appears to date from 1147.",
"All these, however, have been superseded for the modern student by the editions of Natalis de Wailly (1872 and 1874), in which the text is critically edited from all the available MSS.",
"Thus, by degrees, the reproduction of the original text became of secondary importance, and merely served as a pretext for the discussion of topics that had little or no bearing on the context.",
"The method, by which the text was thus utilized as a vehicle for conveying homiletic discourses, traditional sayings, legends and allegories, is abundantly illustrated by the Palestinian and later Targums, as opposed to the more sober translations of Onkelos and the Targum to the Prophets.",
"It would, however, be incorrect to suppose that the translation of the text was left entirely to the individual taste of the translator.",
"The official recognition of a written Targum, and therefore the final fixing of its text belongs to the post-Talmudic period, and is not to be placed earlier than the 5th century.",
"The Hebrew text used by the translators appears to have been practically identical with the Massoretic. The version was held in high esteem in Babylon, and, later, in Palestine, and a special Massora was made for it.",
"On the other hand, they regarded it as necessary to present the sacred text in such a manner as best to convey the particular form of interpretation then current.",
"On the other hand, the version of Onkelos affords just the supplementary material that is required to restore sense to the shorter text.",
"It is not, however, a revision of the Fragmentary Targum - for it is clearly independent of that version - but is rather a parallel, if somewhat later, production, in which the text of Onkelos is already combined with a number of variants and additions.",
"It exhibits, to a marked degree, that tendency to expand the text by additions of every kind, which has been already noted as characteristic of the later stages of Targumic composition.",
"Some writers, notably Professor Zahn,' piecing together this text with 2 Tim.",
"His Latin text is probably as ancient as the Greek text of Marcellus, because the Roman Church must always have been bilingual in its early days.",
"From one of these monasteries the received text seems to have been taken to Rome.",
"We can trace the use of the received text along the line of the journeys both of Pirminius and Boniface, and there is little doubt that they received it from the Roman Church, with which Boniface was in frequent communication.",
"It seems clear, therefore, that the received text was either made or accepted in Rome, c. A.D.",
"At the end of the 8th century Charlemagne inquired of the bishops of his empire as to current forms. The reply of Amalarius of Trier is important because it shows that he not only used the received text, but also connected it with the Roman order of Baptism.",
"This theory, however, depended upon unverified assumptions, such as the supposed silence of theologians about the creed at the beginning of the 9th century; the suggestion that the completed creed would have been useful to them if they had known it as a weapon against the heresy of Adoptianism; the assertion that no MS. containing the complete text was of earlier date than c. 813.",
"Ommanney, who was successful in the discovery of new documents, notably early commentaries, which contained the text of the creed embedded in them, and thus supplied independent testimony to the fact that the creed was becoming fairly widely known at the end of the 8th century.",
"Bethlehemitica; a revised text in 1678 as Synodus Jerosolymitana; Hardouin, Acta conciliorum, vol.",
"At Rome were published the Gospels (with a dedication to Pope Damasus, an explanatory introduction, and the canons of Eusebius), the rest of the New Testament and the version of the Psalms from the Septuagint known as the Psalterium romanum, which was followed (c. 388) by the Psalterium gallicanum, based on the Hexaplar Greek text.",
"Here he did most of his literary work and, throwing aside his unfinished plan of a translation from Origen's Hexaplar text, translated the Old Testament directly from the Hebrew, with the aid of Jewish scholars.",
"The result of all this labour was the Latin translation of the Scriptures which, in spite of much opposition from the more conservative party in the church, afterwards became the Vulgate or authorized version; but the Vulgate as we have it now is not exactly Jerome's Vulgate, for it suffered a good deal from changes made under the influence of the older translations; the text became very corrupt during the middle ages, and in particular all the Apocrypha, except Tobit and Judith, which Jerome translated from the Chaldee, were added from the older versions.",
"But when a committee of the Royal Asiatic Society, with George Grote at its head, decided that the translations of an Assyrian text made independently by the scholars just named were at once perfectly intelligible and closely in accord with one another, scepticism was silenced, and the new science was admitted to have made good its claims. Naturally the early investigators did not fathom all the niceties of the language, and the work of grammatical investigation has gone on continuously under the auspices of a constantly growing band of workers.",
"These were followed by a mediocre edition of the Arabic text of Edrisi's Description of Spain (1799), with notes and a translation.",
"As a Hebrew scholar he made a special study of the history of the Hebrew text, which led him to the conclusion that the vowel points and accents are not an original part of the Hebrew language, but were inserted by the Massorete Jews of Tiberias, not earlier than the 5th century A.D., and that the primitive Hebrew characters are those now known as the Samaritan, while the square characters are Aramaic and were substituted for the more ancient at the time of the captivity.",
"In his hands the history of Florence became a text on which at fitting seasons to deliver lessons in the science he initiated.",
"The two canonical books entitled Ezra and Nehemiah in the English Bible' correspond to the I and 2 Esdras of the Vulgate, to the 2 Esdras of the Septuagint, and to the Ezra and Nehemiah of the Massoretic (Hebrew) text.",
"Here the recension in 1 Esdras especially merits attention for its text, literary structure and for its variant traditions.",
"He attended lectures on grammar, and his favourite work was St Augustine's De civitate Dei, He caused Frankish sagas to be collected, began a grammar of his native tongue, and spent some of his last hours in correcting a text of the Vulgate.",
"There are obvious points of similarity, possibly of derivation, between the details in our text and the above myths, but the subject cannot be further pursued here, save that we remark that in the sun myth the dragon tries to kill the mother before the child's birth, whereas in our text it is after his birth, and that neither in the Egyptian nor in the Greek myth is there any mention of the flight into the wilderness.",
"On the other hand, if we refuse to accept this identification, and hold that the beast from the abyss is yet to come, any attempt at a strict exegesis of the text plunges us in hopeless difficulties.",
"Juvenal, in his seventeenth satire, takes as his text a religious riot between the Tentyrites and the neighbouring Ombites, in the course of which an unlucky Ombite was torn to pieces and devoured by the opposite party.",
"In that manner his influence, as represented by the text of many a statute regulating the relations between Austria and Hungary, is one of an abiding character.",
"Many of these were not pure Shakespeare; and he is credited with the addition of a dying speech to the text of Macbeth.",
"Round the circlet is the singularly inappropriate text from Psalm li., Miserere mei Deus secundum magnam misericordiam tuam.",
"In constituting the text, he imposed upon himself the singular restriction of not inserting any various reading which had not already been printed in some preceding edition of the Greek text.",
"From this rule, however, he deviated in the case of the Apocalypse, where, owing to the corrupt state of the text, he felt himself at liberty to introduce certain readings on manuscript authority.",
"Etienne's division into verses was retained in the inner margin, but the text was divided into paragraphs.",
"The text was followed by a critical apparatus, the first part of which consisted of an introduction to the criticism of the New Testament, in the thirty-fourth section of which he laid down and explained his celebrated canon, Proclivi scriptioni praestat ardua ( The difficult reading is to be preferred to that which is easy), the soundness of which, as a general principle, has been recognized by succeeding critics.",
"His investigations had led him to see that a certain affinity or resemblance existed amongst many of the authorities for the Greek text - MSS., versions, and ecclesiastical writers; that if a peculiar reading, e.g., was found in one of these, it was generally found also in the other members of the same class; and this general relationship seemed to point ultimately to a common origin for all the authorities which presented such peculiarities.",
"Griesbach, and worked up into an elaborate system by the latter critic. Bengel's labours on the text of the Greek Testament were received with great disfavour in many quarters.",
"In answer to these strictures, Bengel published a Defence of the Greek Text of His New Testament, which he prefixed to his Harmony of the Four Gospels, published in 1736, and which contained a sufficient answer to the complaints, especially of Wetstein, which had been made against him from so many different quarters.",
"The text of Bengel long enjoyed a high reputation among scholars, and was frequently reprinted.",
"It approaches the text of Phaedrus so closely that it was probably made directly from it.",
"The reports of work done in this province for several years past form a library of text and illustration.",
"The current Hebrew Text has the land of ammo,i.e.",
"The Greek Text of the New Testament adopted by the Revisers was edited for the Clarendon Press by Archdeacon Palmer (Oxford, 1881).",
"There seems to be no motive sufficient to explain the additions that have been made to the text of the Gospels.",
"Bensly found a complete Syriac text in a MS. recently obtained by the University library at Cambridge.",
"This must remain the standard edition, notwithstanding Dom Morin's most interesting discovery of a Latin version (1894), which was probably made in the 3rd century, and is a valuable addition to the authorities for the text.",
"The Aldine (Venice, 1516) was unfortunately based on a very corrupt MS. The first substantial improvements in the text were due to Casaubon (Geneva, 1587; Paris, 1620), whose text remained the basis of subsequent editions till that of Coraes (Paris, 1815-1819), who removed many corruptions.",
"The collection, in its present form, contains 126 pieces of verse, long and short; that is the number included in the recension of al-Anbari, who had the text from Abu `Ikrima of Dabba, who read it with Ibn al-A`rabi, the stepson and inheritor of the tradition of al-Mufaddal.",
"It is noticeable that this traditional text, and the accompanying scholia, as represented by al-Anbari's recension, are wholly due to the scholars of Kufa, to which place al-Mufaddal himself belonged.",
"The four principal ones have been published for the Pali Text Society, and some volumes have been translated into English or German.",
"Both add notes and explanations of their own, and both have in turn formed the text of commentaries.",
"The authorities for the Crusades have been collected in Bongars, Gesta Dei per Francos (Hanover, 1611) (incomplete); Michaud, Bibliotheque des croisades (Paris, 1829) (containing translations of select passages in the authorities); the Recueil des historiens des croisades, published by the Academie des Inscriptions (Paris, 1841 onwards) (the best general collection, containing many of the Latin, Greek, Arabic and Armenian authorities, and also the text of the assizes; but sometimes poorly edited and still .incomplete); and the publications of the Societe de l'Orient Latin (founded in 1875), especially the Archives, of which two volumes were published in 1881 and 1884, and the volumes of the Revue, published yearly from 1893 to 1902, and containing not only new texts, but articles and reviews of books which are of great service.",
"While the translation was still in progress Ficino from time to time submitted its pages to the scholars, Angelo Poliziano, Cristoforo Landino, Demetrios Chalchondylas and others; and since these men were all members of the Platonic Academy, there can be no doubt that the discussions raised upon the text and Latin version greatly served to promote the purpose of Cosimo's foundation.",
"Nay more, the evidence of the text, so far as it goes, is against such a view.",
"Parthey's, Berlin, 1870, best as to text.",
"Dogma is the whole text of the Bible, doctrinal, historical, scientific, or what not.",
"This was revised in 1537 by Heusbach, and accompanies the Greek text of Herodotus in many editions.",
"The text edited by Montet, 4to (1887).",
"And behind these questions is the fundamental problem of the text, which has been somewhat too slightly treated.",
"The text of Hosea may be in a much worse condition, but a keen scrutiny discloses many an uncertainty, not to say impossibility, in the traditional form of Amos.",
"That the text has been much adapted and altered is certain; not less obvious are the corruptions due to carelessness and accident.",
"Bib.,  Amos, and the introduction to Robertson Smith's Prophets of Israel (2), though in some cases the final decision will have to be preceded by a more thorough examination of the traditional text.",
"The Latin text, together with later recensions and a Greek version, is published in Texts and Studies, i.",
"It is rarely mentioned in Roman history and often confused with Lanuvium or Lanivium in the text both of authors and of inscriptions.",
"The text of the poem is preserved in the Asloan and Bannatyne MSS.",
"There is as yet no satisfactory text of the Rule, either critical or manual; the best manual text is Schmidt's editio minor (Regensburg, 1892).",
"Eichhorn in favour of the borrowing hypothesis of the origin of the synoptical gospels, maintaining the priority of Matthew, the present Greek text having been the original.",
"It is specially valuable in the portion relating to the history of the text (which up to the middle of the 3rd century he holds to have been current only in a common edition (Kocvi EK60cn), of which recensions were afterwards made by Hesychius, an Egyptian bishop, by Lucian of Antioch, and by Origen) and in its discussion of the ancient versions.",
"He plays off the sects against the Catholic Church, the primitive age against the present, Christ against the apostles, the various revisions of the Bible against the trustworthiness of the text and so forth, though he admits that everything was not really so bad at first as it is at present.",
"In addition to such larger insertions, the text of the original document seems to have undergone a certain amount of revision.",
"Unluckily, Mlle de Gournay's original does not appear to exist and her text was said, until the appearance of MM.",
"The Feuillants copy is in existence, being the only manuscript, or partly manuscript, authority for the text; but access to it and reproduction of it are subjected to rather unfortunate restrictions by the authorities, and until it is completely edited students are rather at the mercy of those who have actually consulted it.",
"As early as 1844 he published an edition of the Book of the Revelation, with the Greek text so revised as to rest almost entirely upon ancient evidence.",
"Marr of the Grusian (Georgian) text, and he added to it (Leipzig, 1904) a translation of various small exegetical pieces, which are preserved in a Georgian version only (The Blessing of Jacob, The Blessing of Moses, The Narrative of David and Goliath).",
"This text was published in 1804 by Sir Walter Scott, and was by him assigned to the Rhymer.",
"Of his most important edition, that of the Greek text of the New Testament, something will be said farther on.",
"He edited many authors, it is true, but he had neither the means of forming a text, nor did he attempt to do so.",
"In four reprints, 1519, 1522, 1527, 1 535, Erasmus gradually weeded out many of the typographical errors of his first edition, but the text remained essentially such as he had first printed it.",
"The Greek text indeed was only a part of his scheme.",
"The first forms the text of the principal argument in the Epistle to the Hebrews, in which the author easily demonstrates the inadequacy of the mediation and atoning rites of the Old Testament, and builds upon this demonstration the doctrine of the effectual high-priesthood of Christ, who, in his sacrifice of himself, truly  led His people to God, not leaving them outside as He entered the heavenly sanctuary, but taking them with Him into spiritual nearness to the throne of grace.",
"Becoming a Congregationalist, he accepted in 1842 the chair of biblical criticism, literature and oriental languages at the Lancashire Independent College at Manchester; but he was obliged to resign in 1857, being brought into collision with the college authorities by the publication of an introduction to the Old Testament entitled The Text of the Old Testament, and the Interpretation of the Bible, written for a new edition of Horne's Introduction to the Sacred Scripture.",
"G 2 denotes the Greek text from which the Slavonic and the second Latin Version (consisting of vi.-xi.) were translated.",
"Part of the original work omitted by the final editor of our book is preserved in the Opus imperfectum, which goes back not to our text, but to the original Martyrdom.",
"These he reduced or enlarged as it suited his purpose, and put them together as they stand in our text.",
"The later recension of this Vision was used by Jerome, and a more primitive form of the text by the Archontici according to Epiphanius.",
"This translation is made from Charles's text, and his analysis of the text is in the main accepted by this scholar.",
"He was a diligent student of Shakespeare, and his last literary work was On the Received Text of Shakespeare's Dramatic Writings and its Improvement (1862).",
"The date of the manuscript appears to be the middle of the 14th century, and probably in its present form it is only a copy of a much older text; there is also a translation of the fiftieth psalm belonging to the 13th century.'",
"About ten years after the death of Sidonius we find Asterius, the consul of 494, critically revising the text of Virgil in Rome.",
"Among the scholars of Italian birth, probably the only one in this age who rivalled the Greeks as a public expositor of their own literature was Politian (1454-1494), who lectured on Homer and Aristotle in Florence, translated Herodian, and was specially interested in the Latin authors of the Silver Age and in the text of the Pandects of Justinian.",
"A comparatively subordinate place was assigned to Greek, especially as the importance attributed to the Vulgate weakened the motive for studying the original text.",
"It is probable that the present text became fixed as early as the 2nd century A.D., but even this earlier date leaves a long interval between the original autographs of the Old Testament writers and our present text.",
"Since the fixing of the Massoretic text the task of preserving and transmitting the sacred books has been carried out with the greatest care and fidelity, with the result that the text has undergone practically no change of any real importance; but before that date, owing to various causes, it is beyond dispute that a large number of corruptions were introduced into the Hebrew text.",
"In dealing, therefore, with the textual criticism of the Old Testament it is necessary to determine the period at which the text assumed its present fixed form before considering the means at our disposal for 'controlling the text when it was, so to speak, in a less settled condition.",
"More important are those passages in which the Massoretes have definitely adopted a variation from the consonantal text.",
"Many even of these readings merely relate to variations of spelling, pronunciation or grammatical forms; others substitute a more decent expression for the coarser phrase of the text, but in some instances the suggested reading really affects the sense of the passage.",
"They do not point to any critical editing of the text; for the aim of the Massoretes was essentially conservative.",
"Their object was not to create a new text, but rather to ensure the accurate transmission of the traditional text which they themselves had received.",
"Their work may be said to culminate in the vocalized text which resulted from the labours of Rabbi Aaron ben Asher in the 10th century.",
"Hence it is hardly doubtful that the form in which we now possess the Hebrew text was already fixed by the beginning of the 2nd century.",
"On the other hand, evidence such as that of the Book of Jubilees shows that the form of the text still fluctuated considerably as late as the 1st century A.D., so that we are forced to place the fixing of the text some time between the fall of Jerusalem and the production of Aquila's version.",
"The latter's system of interpretation was based upon an extremely literal treatment of the text, according to which the smallest words or particles, and sometimes even the letters of scripture, were invested with divine authority.",
"The inevitable result of such a system must have been the fixing of an officially recognized text, which could scarcely have differed materially from that which was finally adopted by the Massoretes.",
"But if the evidence available points to the time of Hadrian as the period at which the Hebrew text assumed its present form, it is even more certain that prior to that date the various MSS.",
"Indications also are not wanting in the Hebrew text itself to show that in earlier times the text was treated with considerable freedom.",
"Thus, according to Jewish tradition, there are eighteen7 passages in which the older scribes deliberately altered the text on the ground that the language employed was either irreverent or liable to misconception.",
"These intentional alterations, however, only affect a very limited portion of the text, and, though it is possible that other changes were introduced at different times, it is very This represents the Western tradition as opposed to the Eastern text of ben Naphtali.",
"For these Tigqune Sopherim or  corrections of the scribes  see Geiger, Urschrift, pp. 308 f.; Strack, Prolegomena Critica, p. 87; Buhl, Canon and Text of the Old Testament, pp. 103 f.",
"Less important are the Itture Sopherim, or five passages in which the scribes have omitted a waw from the text.",
"Externally also the ancient versions, especially the Septuagint, frequently exhibit variations from the Hebrew which are not only intrinsically more probable, but often explain the difficulties presented by the Massoretic text.",
"In the use of the ancient versions for the purposes of textual criticism there are three precautions which must always be observed; we must reasonably assure ourselves that we possess the version itself in its original integrity; we must eliminate such variants as have the appearance of originating merely with the translator; the remainder, which will be those that are due to a difference of text in the MS. (or MSS.) used by the translator, we must then compare carefully, in the light of the considerations just stated, with the existing Hebrew text, in order to determine on which side the superiority lies.",
"The Samaritan Pentateuch agrees with the Septuagint version in many passages, but its chief importance lies in the proof which it affords as to the substantial agreement of our present text of the Pentateuch, apart from certain intentional changes,' with that which was promulgated by Ezra.",
"Its value for critical purposes is considerably discounted by the late date of the MSS., upon which the printed text is based.",
"The text which they exhibit is virtually identical with the Massoretic text.",
"The opposition, as might be expected, came from the side of the Jews, and was due partly to the controversial use which was made of the version by the S Christians, but chiefly to the fact that it was not suffi- ciently in agreement with the standard Hebrew text estab.- lished by Rabbi Aqiba and his school.",
"That Origen did not succeed in his object of recovering the original Septuagint is due to the fact that he started with the false conception that the original text of the Septuagint must be that which coincided most nearly with the current Hebrew text.",
"For the Hexaplar text which he thus produced not only effaced many of the most characteristic features of the old version, but also exercised a prejudicial influence on the MSS.",
"The Hexapla as a whole was far too large to be copied, but the revised Septuagint text was published separately by Eusebius and Pamphilus, and was extensively used in Palestine during the 4th century.",
"His revision (to quote Dr Swete)  was doubtless an attempt to revise the (or ' common text ' of the Septuagint) in accordance with the principles of criticism which were accepted at Antioch.",
"To Ceriani is due the discovery that the text preserved by codices 19, 82, 93, 108, really represents Lucian's recension; the same conclusion was reached independently by Lagarde, who combined codex 118 with the four mentioned above.",
"It may be noted here that the Complutensian Polyglott represents a Lucianic text.",
"Of the other books which he revised according to the Hexaplar text, that of Job has alone come down to us.",
"For textual purposes the Vulgate possesses but little value, since it presupposes a Hebrew original practically identical with the text stereotyped by the Massoretes.",
"Its value for textual purposes is not great, partly because the underlying text is the same as the; Massoretic, partly because the Syriac text has at different times been harmonized with that of the Septuagint.",
"Text des A.T.",
"It then becomes the task of critical exegesis to interpret the text thus recovered so as to bring out the meaning intended by the original authors.",
"Jewish study was exclusively based on the official Hebrew text, which was fixed, probably in the 2nd century A.D., and thereafter scrupulously preserved.",
"This text, however, had suffered certain now obvious corruptions, and, probably enough, more corruption than can now, or perhaps, ever will be, detected with certainty.",
"Beneath the ancient Greek version, the Septuagint, there certainly underlay an earlier form of the Hebrew text than that perpetuated by Jewish tradition, and if Christian scholars could have worked through the version to the underlying Hebrew text, they would often have come nearer to the original meaning than their Jewish contemporaries.",
"But this they could not do; and since the version, owing to the limitations of the translators, departs widely from the sense of the original, Christian scholars were on the whole kept much farther from the original meaning than their Jewish contemporaries, who used the Hebrew text; and later, after Jewish grammatical and philological study had been stimulated by intercourse with the Arabs, the relative disadvantages under which Christian scholarship laboured increased.",
"Jerome, perceiving the unsatisfactory position of Latin-speaking Christian scholars who studied the Old Testament at a double remove from the original - in Latin versions of the Greek - made a fresh Latin translation direct from the Hebrew text then received among the Jews.",
"Subsequently, however, this version of Jerome (the Vulgate) became the basis of Western Biblical scholarship. Henceforward the Western Church suffered both from the corruptions in the official Hebrew text and also from the fact that it worked from a version and not from the original, for a.",
"But if the use of versions, or of an uncritical text of the original, was one condition unfavourable to criticism, another that was not less serious was the dominance over both Jews and Christians of unsound methods of interpretation - legal or dogmatic or allegorical.",
"Yet even so the publication of the Hebrew text by Christian scholars marks an important stage; henceforth the study of the original enters increasingly into Christian Biblical scholarship; it already underlay the translations which form so striking a feature of the 16th century.",
"There are also, however, certain conditions peculiar to the text of the Old Testament.",
"But the fact that the later text makes use of the earlier Ito make itself intelligible in no way destroys the fact that it is .as entirely distinct a work from the earlier as is any commentary distinct from the work on which it comments.",
"The first task, of Old Testament textual criticism after the Reformation was to prove the independence of these two texts, to gain general Tecognition of the fact that vowels and accents formed no part .of the original Hebrew text of the Old Testament.",
"The coeval origin of consonants and vowels had indeed been questioned or denied by the earliest reformers (Luther, Zwingli, Calvin), but later, in the period of Protestant scholasticism and under the influence of one school of Jewish Rabbis, Protestant scholars in particular, and especially those .of the Swiss school, notably the Buxtorfs, had committed themselves to the view that the vowels formed an integral and original part of the text of the Old Testament; and this they maintained with all the more fervency.",
"But the original text of the Old Testament long before it was combined with the text of the Jewish or Massoretic interpretation had already undergone a somewhat similar change, the extent of which was indeed far less, but also less clearly discoverable.",
"For reasons suggested partly by the study of Semitic inscriptions, partly by comparison of passages occurring twice within the Old Testament, and partly by a comparison of the Hebrew text with the Septuagint, it is clear that the authors of the Old Testament (or at least most of them) themselves made some use of these vowel consonants, but that in a great number of cases the vowel consonants that stand in our present text were inserted by transcribers and editors of the texts.",
"In view of all this, the first requisite for a critical treatment of the text of the Old Testament is to consider the consonants by themselves, to treat every vowel-consonant as possibly not original, and the existing divisions of the text into words as original only in those cases where they yield a sense better than any other possible division (or, at least, as good).",
"Apart from these changes in the history of the text, it has, like all ancient texts, suffered from accidents of transmission, from the unintentional mistakes of copyists.",
"This fact was, naturally enough and under the same dogmatic stress, denied by those scholars who maintained that the vowels were an integral part of the text.",
"Capellus drew conclusions from such important facts as the occurrence of variations in the two Hebrew texts of passages found twice in the Old Testament itself, and the variations brought to light by a comparison of the Jewish and Samaritan texts of the Pentateuch, the Hebrew text and the Septuagint, the Hebrew text and New Testament quotations from the Old Testament.",
"In order that the principles already perceived by Capellus might be satisfactorily applied in establishing a critical text, many things were needed; for example, a complete collation of existing MSS.",
"These editions furnish the material, but neither attempts the actual construction of a critical text of the version.",
"Some important contributions towards a right critical method of using the material collected have been made - in particular by Lagarde, who has also opened up a valuable line of critical work, along which much remains to be done, by his restoration of the Lucianic recension, one of the three great recensions of the Greek text of the Old Testament which obtained currency at the close of the 3rd and beginning of the 4th centuries A.D.",
"More especially since the time of Capellus the value of the Septuagint for correcting the Hebrew text has been recognized; but it has often been used uncritically, and the correctness of the Hebrew text underlying it in comparison with the text of the Hebrew MSS., though still perhaps most generally underestimated, has certainly at times been exaggerated.",
"It has only been possible here to indicate in the briefest way what is involved in the collection and critical sifting of the extant evidence for the text of the Old Testament, Results of how much of the work has been done and how much Criticism.",
"In so far as it is possible to recover the Hebrew text from which the Greek version was made, it is possible to recover a form of the Hebrew text current about 280 B.C. in the case of the Pentateuch, some time before loo B.C. in the case of most of the rest of the Old Testament.",
"By a comparison of these two lines of evidence we can approximate to a text current about 300 B.C. or later; but for any errors which had entered into the common source of these two forms of the text we possess no documentary means of detection whatsoever.",
"Cheyne (in Critica Biblica, 1903), whose restorations resting on a dubious theory of Hebrew history have met with little approval, though his negative criticism of the text is often keen and suggestive.",
"Haupt's Sacred Books of the Old Testament, edited by various scholars, was designed to present, when complete, a critical text of the entire Old Testament with critical notes.",
"The results of textual criticism, including a considerable number of conjectural emendations, are succinctly presented in Kittel's Biblia Hebraica (1906); but the text here printed is the ordinary Massoretic (vocalized) text.",
"It is also to be noted that in the Samaritan text of the Pentateuch, and in the LXX., the figures, especially in the period from the Creation to the birth of Abraham, differ considerably from those given in the Hebrew, yielding in Sam.",
"The phenomena which are sometimes supposed to require the hypothesis of an Ur-Marcus are more simply and satisfactorily explained as incidents in the transmission of the Marcan text.",
"The list recognized four Gospels, Acts, thirteen epistles of Paul, two epistles of John, Jude, Apocalypse of John and (as the text stands) of Peter; there is no mention of Hebrews or (apparently) of 3 John or Epistles of Peter, where it is possible - we cannot say more - that the silence as to t Peter is accidental; the Shepherd of Hermas on account of its date is admitted to private, but not public, reading; various writings associated with Marcion, Valentinus, Basilides and Montanus are condemned.",
"At present it has not seriously threatened the hold of Gregory's notation on the critical world, but it will probably have to be adopted, at least to a large extent, when von Soden's text is published.",
"The text has been corrected by two scribes, one (the S copOw,r) contemporary with the original writer, the other belonging to the 10th or 11th century.",
"The text is the best example of the so-called Neutral Text, except in the Pauline epistles, where it has a strong  Western  element.",
"The text was written, according to Tischendorf, by four scribes, of whom he identified one as also the scribe of cod.",
"It has, in the main, a Neutral text, less mixed in the Epistles than that of B, but not so pure in the Gospels.",
"The text is written in one column to a page, the Greek on the left hand page and the Latin on the right.",
"The text of this MS. is important as the oldest and best witness in a Greek MS. to the so-called  Western  text.",
"Its text in the Old Testament is thought by some scholars to show signs of representing the Hesychian recension, but this view seems latterly to have lost favour with students of the Septuagint.",
"In the New Testament it has in the gospels a late text of Westcott and Hort's  Syrian  type, but in the epistles there is a strongly marked  Alexandrian  element.",
"Hort (Intro- duction, p. 268) has shown from a consideration of displacements in the text of the Apocalypse that it was copied from a very small MS., but this, of course, only holds good of the Apocalypse.",
"The character of the text is mixed with a strong  Alexandrian  element.",
"It is probably the best extant witness to the type of Greek text which was in use in Italy at an early time.",
"They have a most peculiar text of a mainly  Western  type, with some special affinities to the Old Syriac and perhaps to the Diatessaron.",
"They are known as the Ferrar group in memory of the scholar who first published their text, and are sometimes quoted as 4, (which, however, properly is the symbol for Codex Beratinus of the Gospels), and sometimes as fam.13.",
"It is best discussed by Rendel Harris's books, The Origin of the Leicester Codex (1887), The Origin of the Ferrar Group (1893), and The Ferrar Group (1900), all published at Cambridge; the text of fam.'",
"In the Acts the European text is found in cod.",
"To remedy the confusion produced by the variations of the Latin text Pope Damasus asked Jerome to undertake a revision, and the latter published a new text of the New Testament in A.D.",
"This version gradually became accepted as the standard text, and after a time was called the  Vulgata, the first to use this name as a title being, it is said, Roger Bacon.",
"This type of text he revised with the help of Greek MSS.",
"Of Jerome's revision we possess at least 8000 MSS., of which the earliest may be divided (in the gospels at all events) into groups connected with various countries; the most important are the Northumbrian, Irish, Anglo-Irish and Spanish, but the first named might also be called the Italian, as it represents the text of good MSS.",
"The text soon began to deteriorate by admixture with the Old Latin, as well from the process of transcription, and several attempts at a revision were made before the invention of printing.",
"White probably restores the text almost to the state in which Jerome left it.",
"It seems certain that the Old Syriac version also contained the Acts and Pauline epistles, as Aphraates and Ephraem agree in quoting a text which differs from the Peshito, but no MSS.",
"C. Burkitt that the portion contain ing the gospels was made by Rabbula, bishop of Edessa (411), to take the place of the Diatessaron, and was based on the Greek text which was at that time in current use at Antioch.",
"But he never added 2 Peter, Jude, 2 and 3 John, or the Apocalypse, and the text of these books, which is sometimes bound up with the Peshito, really is that of the Philoxenian or of the Harklean version.",
"A comparison of the Peshito with quotations in Aphraates and Ephraem shows that Rabbula revised the text of the Acts and Pauline epistles, but in the absence of MSS.",
"It represents in the main the text of the later Greek MSS., but it has important textual notes, and has adopted a system of asterisks and obeli from the Hexaplar LXX.",
"The marginal readings are therefore valuable evidence for the Old Alexandrian text.",
"White at Oxford under the title Versio Philoxenia; for the marginal notes see esp. Westcott and Hort, Introduction, and for Acts, Pott's Abendldndische Text der Apostelgesch.",
"There are now but few, if any, scholars who think that the Peshito is an entirely separate version, and the majority have been convinced by Burkitt and recognize (1) that the Peshito is based on a knowledge of the Old Syriac and the Diatessaron; (2) that it was made by Rabbula with the help of the contemporary Greek text of the Antiochene Church.",
"Here it is necessary to distinguish between the original text of the Old Syriac and the existing MSS.",
"Horner's researches tend to show that the Greek text on which it was based was different from that represented by the Bohairic, and probably was akin to the  Western  text, perhaps of the type used by Clement of Alexandria.",
"Its limitations are found in the inaccuracy of quotation of the writers, and often in the corrupt condition of their text.",
"This latter point especially affects quotations which later scribes frequently forced into accord with the text they preferred.",
"None of these groups bears witness to quite the same text, nor can all of them be identified with the texts found in existing MSS.",
"In group 4 the situation is more complex; Clement used a text which has most in common with.",
"Group 4 has a peculiar text which cannot be identified with any definite group of MSS.",
"The problem which faces the textual critic of the New Testament is to reconstruct the original text from the materials supplied by the MSS., versions, and quotations in early writers, which have been described in the preceding section on the apparatus criticus.",
"His object, therefore, is to discover and remove the various corruptions which have crept into the text, by the usual methods of the textual critic - the collection of material, the grouping of MSS.",
"But in practice it is general, and certainly convenient, to regard their work rather as material for criticism, and to begin the history of textual criticism with the earliest printed editions which sought to establish a standard Greek Text.",
"It was printed in 1514, and is thus the first printed text, but is not the first published, as it was not issued until 1522.",
"Ximenes used, but it is plain from the character of the text that they were not of great value.",
"His text was reprinted in 1569 by Chr.",
"The first published text was that of Erasmus.",
"These are 16mo volumes, but the third and most important edition (1550) was a folio with a revised text.",
"It is this edition which is usually referred to as the text of Stephanus.",
"A fourth edition (in 16mo) published at Geneva in 1551 is remarkable for giving the division of the text into verses which has since been generally adopted.",
"They did not greatly differ from the 1550 edition of Stephanus, but historically are important for the great part they played in spreading a knowledge of the Greek text, and as supplying the text which the Elzevirs made the standard on the continent.",
"The two brothers, Bonaventura and Abraham Elzevir, published two editions at Leiden in 1624 and 1633, based chiefly on Beza's text.",
"The Elzevir text has formed the basis of all non-critical editions on the continent, but in England the 1550 edition of Stephanus has been more generally followed.",
"The importance of both the Stephanus and Elzevir editions is that they formed a definite text for the purposes of comparison, and so prepared the way for the next stage, in which scholars busied themselves with the investigation and collation of other MSS.",
"The first to begin this work was Brian Walton, bishop of Chester, who published in 1657 in the 5th and 6th volumes of his  polyglot  Bible the text of Stephanus (1550) with the readings of fifteen new MSS.",
"In 1675 John Fell, dean of Christ Church, published the Elzevir text with an enlarged apparatus, but even more important was the help and advice which he gave to the next important editor - Mill.",
"It gives the text of Stephanus (1550) with collations of 78 MSS., besides those of Stephanus, the readings of the Old Latin, so far as was then known, the Vulgate and Peshito, together with full and valuable prolegomena.",
"A little later Richard Bentley conceived the idea that it would be possible to reconstruct the original text of the New Testament by a comparison of the earliest Greek and Latin sources; he began to collect material for this purpose, and issued a scheme entitled  Proposals for Printing  in 1720, but though he amassed many notes nothing was ever printed.",
"Wetstein, one of Bentley's assistants, when living in Basel in 1730, published  Prolegomena  to the Text, and in 1751-1752 (at Amsterdam) the text of Stephanus with enlarged Prolegomena and apparatus criticus.",
"His view was that the last group was the least valuable; but, except when internal evidence forbade (and he thought that it frequently did so), he followed the text found in any two groups against the third.",
"In this great book a break was made for the first time with the traditional text and the evidence of the late MSS., and an attempt was made to reconstruct the text according to the oldest authorities.",
"As a collector and publisher of evidence Tischendorf was marvellous, but as an editor of the text he added little to the principles of Lachmann, and like Lachmann does not seem to have appreciated the value of the Griesbachian system of grouping MSS.",
"They finally exploded the pretensions of the Textus Receptus to be the original text; but neither of them gave any explanation of the relations of the later text to the earlier, nor developed Griesbach's system of dealing with groups of MSS.",
"Hort (commonly quoted as WH), the Cambridge scholars, supplied the deficiencies of Lachmann, and without giving up the advantages of his system, and its development by Tischendorf, brought back the study of the text of the New Testament to the methods of Griesbach.",
"Their view of the history of the text is that a comparison of the evidence shows that, while we can distinguish more than one type of text, the most clearly discernible of all the varieties is first recognizable in the quotations of Chrysostom, and is preserved in almost all the later MSS.",
"Though found in so great a number of witnesses, this type of text is shown not to be the earliest or best by the evidence of all the oldest MS. versions and Fathers, as well as by internal evidence.",
"It is impossible, in face of the fact that the evidence of the oldest witnesses of all sorts is constantly opposed to the longer readings, to doubt that WH were right in arguing that these phenomena prove that the later text was made up by a process of revision and conflation of the earlier forms. Influenced by the use of the later text by Chrysostom, WH called it the Syrian or Antiochene text, and refer to the revision which produced it as the Syrian revision.",
"They suggested that it might perhaps be attributed to Lucian, who is known to have made a revision of the text of the LXX.",
"Bezae and in Syr C; (2) the Alexandrine text used by Cyril of Alexandria and found especially in CL 33; and (3) a text which differs from both the above mentioned and is therefore called by WH the Neutral text, found especially in rt B and the quotations of Origen.",
"Their reason was that omission seems to be contrary to the genius of the Western text, and that it is therefore probable that these passages represent interpolations made in the text on the Neutral side after the division between it and the Western.",
"Having thus decided that the Neutral text was almost always right, it only remained for WH to choose between the various authorities which preserved this type.",
"The great importance of this work of WH lies in the facts that it not merely condemns but explains the late Antiochene text, and that it attempts to consider in an objective manner all the existing evidence and to explain it historically and genealogically.",
"The text reached is not widely different from that of WH.",
"Luke wrote the first edition of the Gospel for Theophilus from Caesarea; this is the Neutral text of the Gospel.",
"Afterwards he went to Rome and there revised the text of the Gospel and reissued it for the Church in that city; this is the Western (or, as Blass calls it, Roman) text of the Gospel.",
"At the same time he continued his narrative for the benefit of the Roman Church, and published the Western text of the Acts.",
"Finally he revised the Acts and sent a copy to Theophilus; this is the Neutral text of the Acts.",
"This ingenious theory met with considerable approval when it was first advanced, but it has gradually been seen that  Western  text does not possess the unity which Blass's theory requires it to have.",
"To some extent influenced by and using Bousset's results, Schmidtke has tried to show that certain small lines in the margin of B point to a connexion between that MS. and a Gospel harmony, which, by assuming that the text of B is Hesychian, he identifies with that of Ammonius.",
"Nestle, however, and other scholars think that the lines in B are merely indications of a division of the text into senseparagraphs and have nothing to do with any harmony.",
"Rendel Harris argued for the influence of Latin, and Chase for that of Syriac. While both threw valuable light on obscure points, it seems probable that they exaggerated the extent to which retranslation can be traced; that they ranked Codex Bezae somewhat too highly as the best witness to the  Western  text; and that some of their work was rendered defective by their failure to recognize quite clearly that the  Western  text is not a unity.",
"We now know more about the Old Latin, and, thanks to Mrs Lewis' discovery, much more about the Old Syriac. The result is that the authorities on which WH relied for their Western text are seen to bear witness to two texts, not to one.",
"The Old Latin, if we take the African form as the oldest, as compared with the Neutral text has a series of interpolations and a series of omissions.",
"The Old Syriac, if we take the Sinaitic MS. as the purest form, compared in the same way, has a similar double series of interpolations and omissions, but neither the omissions nor the interpolations are the same in the Old Latin as in the Old Syriac. Such a line of research suggests that instead of being able, as WH thought, to set the Western against the Neutral text (the Alexandrian being merely a development of the latter), we must consider the problem as the comparison of at least three texts, a Western (geographically), an Eastern and the Neutral.",
"More recent investigations have confirmed their view as to the relation of the Alexandrian to the Neutral text, but have thrown doubt on the age and widespread use of the latter.",
"Whatever view be taken of the provenance of Codex Vaticanus it is plain that its archetype had the Pauline epistles in a peculiar order which is only found in Egypt, and so far no one has been able to discover any non-Alexandrian writer who used the Neutral text.",
"Moreover, Barnard's researches into the Biblical text of Clement of Alexandria show that there is reason to doubt whether even in Alexandria the Neutral text was used in the earliest times.",
"We have no evidence earlier than Clement, and the text of the New Testament which he quotes has more in common with the Old Latin or  geographically Western  text than with the Neutral, though it definitely agrees with no known type preserved in MSS.",
"This discovery has put the Neutral text in a different light.",
"It would seem as though we could roughly divide the history of the text in Alexandria into three periods.",
"It thus seems probable that WH's theory must be modified, both as regards the  Western  text, which is seen not to be a single text at all, and as regards the  Neutral  text, which seems to be nothing more than the second stage of the development of the text in Alexandria.",
"Both of these point to the existence in the 3rd and even 2nd century of types of text which differ in very many points from anything preserved in Greek MSS.",
"The question, therefore, is whether we ought not to base our text on the versions and ecclesiastical quotations rather than on the extant Greek MSS.",
"It is also possible to argue, as WH did, on the same side, that the purest form of text was preserved in Alexandria, from which the oldest uncials are directly or indirectly derived, but this argument has been weakened if not finally disposed of by the evidence of Clement of Alexandria.",
"This has not yet been done, but enough has been accomplished to point to the probability that the result will be the establishment of at least three main types of texts, represented by the Old Syriac, the Old Latin and Clement's quotations, while it is doubtful how far Tatian's Diatessaron, the quotations in J ustin and a few other sources may be used to reconstruct the type of Greek text used in Rome in the 2nd century when Rome was still primarily a Greek church.",
"The second stage must be the comparison of these results and the attempt to reconstruct from them a Greek text from which they all arose.",
"The doctrines of Christianity, and in many communities the customs of the Church, were held to be inferences from the inspired text of the Scriptures.",
"The editions of Mill (1707) and of Wetstein (1751) proved once for all that variations in the text, many of them serious, had existed from the earliest times.",
"Often the biblical text cannot be said to supply more than a hint or a suggestion, and the particular application in Halaka or Haggada must be taken on its merits, and the teaching does not necessarily fall because the exegesis is illegitimate.",
"The homiletic Midrashim are characterized by (a) a proem, an introduction based upon some biblical text (not from the lesson itself), which led up to (b) the exposition of the lesson, the first verse of which is more fully discussed than the rest.",
"This would make the Round Table analogous to the turning castles which we frequently meet with in romances; and while explaining the peculiarities of Layamon's text, would make it additionally probable that he was dealing with an earlier tradition of folklore character, a tradition which was probably also familiar to Wace, whose version, though much more condensed than Layamon's, is yet in substantial harmony with this latter.",
"In 797 Charlemagne commissioned Alcuin to prepare an emended text of the Vulgate; copies of this text were multiplied, not always accurately, in the famous writingschools at Tours.",
"New versions are made, wherever practicable, from the original Hebrew or Greek text, and the results thus obtained have a high philological value and interest.",
"The corrupt text in Chronicles of 3000 baths would need a still longer cubit; and, if a lesser cubit of 21.6 or 18 in, be taken, the result for the size of the bath would be impossibly small.",
"Copies of the full text of the Scotichronicon, by different scribes, are extant.",
"Goodall's is the only complete modern edition of Bower's text.",
"It consists of two pages of preface followed by sixty-seven pages of text.",
"Caesar at once marched to meet them, and, on the pre text that they had violated a truce, seized their leaders who had come to parley with him, and then surprised and practically destroyed their host.",
"During this long period he occupied himself with completing the constitutions by incorporating certain declarations, said to be Ignatian, which explained and sometimes completely altered the meaning of the original text.",
"Finally, we should mention in this connexion the text on which are based the pseudo-Clementine Homilies and Recognitiones (beginning of the 3rd century).",
"Another early explorer was the French artist Frederic de Waldeck, who published Voyage pittoresque et archeologique dans la province d'Yucatan (Paris, 1838), and whose collection of drawings appeared in 1866, with the descriptive text by Brasseur de Bourbourg, under the title Monuments anciens du Mexique.",
"He was invited to approve the candidates proposed for state governorships; in all law cases affecting the Government or political matters the judges asked his opinion; he drafted bills, and discussed their text with individual members and committees of congress.",
"The Latin text dates from the close of the 7th century, and is the work of Eadfrith, bishop of Lindisfarne (698-721).",
"The Latin original is a glossed version of the Vulgate, and in the English translation the words of the gloss are often substituted for the strong and picturesque expressions of the Biblical text;.",
"The following brief extracts may exemplify the hermit's rendering and the change the text underwent in later copies.'",
"Several revisions of the text exist, the later of which present such striking agreement with the later Wycliffite version that we shall not be far wrong if we assume that they were made use of to a considerable extent by the revisers of this version.",
"The translation of these Gospels as well as of the Epistles referred to above is stiff and awkward, the translator being evidently afraid of any departure from the Latin text of his original.",
"It was the custom of the medieval preachers and writers to give their own English version of any text which they quoted, not resorting as in later times to a commonly received translation.",
"This explains the fact that in collections of medieval homilies that have come down to us, no two renderings of the Biblical text used are ever alike, not even Wycliffe himself making use of the text of the commonly accepted versions that went under his name.",
"It is noteworthy that these early versions from Anglo-Saxon times onwards were perfectly orthodox, executed by and for good and faithful sons of the church, and, generally speaking, with the object of assisting those whose knowledge of Latin proved too scanty for a proper interpretation and understanding of the holy text.",
"The text of the Gospels was extracted from the Commentary upon them by Wycliffe, and to these were added the Epistles, the Acts and the Apocalypse, all now translated anew.",
"It would appear, however, as if at first at all events the persecution was directed not so much against the Biblical text itself as against the Lollard interpretations which accompanied it.",
"In a convocation held at Oxford under Archbishop Arundel in 1408 it was enacted  that no man hereafter by his own authority translate any text of the Scripture into English or any other tongue, by way of a book, booklet, or tract; and that no man read any such book, booklet, or tract, now lately composed in the time of John Wycliffe or since, or hereafter to be set forth in part or in whole, publicly or privately, upon pain of greater excommunication, until the said translation be approved by the ordinary of the place, or, if the case so require, by the council provincial.",
"For all this, manuscripts of Purvey's Revision were copied and re-copied during this century, the text itself being evidently approved by the ecclesiastical authorities, when in the hands of the right people and if unaccompanied by controversial matter.",
"Erasmus in 1516 published the New Testament in Greek, with a new Latin version of his own; the Hebrew text of the Old Testament had been published as early as 1488.",
"Coverdale consulted in his revision the Latin version of the Old Testament with the Hebrew text by Sebastian Minster, the Vulgate and Erasmus's editions of the Greek text for the New Testament.",
"It was the first Bible which had the text divided into  verses and sections according to the best editions in other languages.",
"It represented in the Old Testament a thorough and independent revision of the text of the Great Bible with the help of the Hebrew original, the Latin versions of Leo Judd (1543), Pagninus (1528), Sebastian Munster (1534-1535), and the French versions of Olivetan.",
"The New Testament consisted of Tyndale's latest text revised to a great extent in accordance with Beza's translation and commentary.",
"To use sections and divisions in the text as Pagnine in his translation useth, and for the verity of the Hebrew to follow the said Pagnine and Munster specially, and generally others learned in the tongues.",
"To make no bitter notes upon any text, or yet to set down any determination in places of controversy.",
"The received Hebrew text has undergone but little emendation, and the revisers had before them substantially the same Massoretic text which was in the hands of the translators of 1611.",
"It was felt that there was no sufficient justification to make any attempt at an entire reconstruction of the text on the authority of the versions.",
"But the advance in the study of Hebrew since the early part of the 17th century enabled them to give a more faithful translation of the received text.",
"The results of modern critical methods could not fail to make the incompleteness of the  Received Text, and of the  Authorized Version, which was based on it, obvious.",
"The revisers' first task was to reconstruct the Greek text, as the necessary foundation of their work.",
"The revisers were privately supplied with instalments of Westcott and Hort's text as their work required them.",
"The changes in the Greek text of the Authorized Version when compared with the textus receptus are numerous, but the contrast between the English versions of 1611 and 1881 are all the more striking because of the difference in the method of translation which was adopted.",
"The text of the Revised Version is printed in paragraphs, the old division of books into chapters and verses being retained for convenience of reference.",
"Ecclesiastical conservatives were scandalized by the freedom with which the traditional text was treated.",
"It is curious that this tradition is ascribed by al-Marzugi and his teacher Abu 'Ali al-Farisi to Abu `Ikrima of Dabba, who is represented by al-Anbari as the transmitter of the correct text from Ibn al-A`rabi.",
"In America there are at Yale University a modern copy of the same recension, taken from the same original as the Cairo copy, and a MS. of Persian origin, dated 1657, presenting a text identical with the Vienna codex.",
"In 1885 Professor Heinrich Thorbeckebegan an edition of the text based on the Berlin codex, but only the first fasciculus, containing forty-two poems, had appeared when his work was cut short by death.",
"In 1891 the first volume of an edition of the text, with a short commentary taken from al-Anbari, was printed at Constantinople.",
"In 1906 an edition of the whole text, with short glosses taken from al-Anbari's commentary, was published at Cairo by Abu Bakr"]


# remove special characters and use lowercase

for i, s in enumerate(sample_text):
    s = remove_special_characters(s)
    s = s.lower()
    sample_text[i] = s

# define parameters
vocab_size = 10000
embedding_dim = 16
max_length = 256
trunc_type='post'
padding_type='post'
oov_tok = "<OOV>"
num_epochs = 500


# word embedding
tokenizer = tf.keras.preprocessing.text.Tokenizer(oov_token=oov_tok, num_words = vocab_size) # define out of vocabulary words token

tokenizer_word.fit_on_texts(sample_text)
print("Showing tokenizers - Words")
print(tokenizer_word.word_index)
print("\n\n\n")

# convert to sequence
train_sequence_words = tokenizer_word.texts_to_sequences(sample_text)

# pad and truncate
padded_train_sequence_words = tf.keras.preprocessing.sequence.pad_sequences(train_sequence_words, padding=padding_type, truncating=trunc_type, maxlen=max_length)

# create embedding
embedding_model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length = max_length),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(1)
])



train_data = np.array(padded_train_sequence_words)

metrics_to_be_used = ['accuracy']

embedding_model.compile(loss='mse',optimizer='adam', metrics=metrics_to_be_used)
history = embedding_model.fit(train_data, train_data, epochs=num_epochs, validation_split=0.2,)

print(embedding_model.summary())

variables_for_plot = ["loss"] + metrics_to_be_used

for var in variables_for_plot:

    loss_train = history.history["{}".format(var)]
    loss_val = history.history['val_{}'.format(var)]
    epochs = range(1,len(history.history['loss'])+1)
    plt.figure()
    plt.plot(epochs, loss_train, 'g', label='Training loss')
    plt.plot(epochs, loss_val, 'b', label='Validation loss')
    plt.title('{}'.format(var))
    plt.xlabel('Epochs')
    plt.ylabel(var)
    plt.legend()
plt.show()


w1 = "independent"
w2 = "Original"
sample = [w1,w2]

def get_embedding_value(sample):

    for i, s in enumerate(sample):
        s = remove_special_characters(s)
        s = s.lower()
        sample[i] = s

    sequence = tokenizer_word.texts_to_sequences(sample)
    padded_sequence = tf.keras.preprocessing.sequence.pad_sequences(sequence,
                                                                                padding=padding_type,
                                                                                truncating=trunc_type,
                                                                                maxlen=max_length)

    output = embedding_model.predict(np.array(padded_sequence))

    return output

output = get_embedding_value(sample)
