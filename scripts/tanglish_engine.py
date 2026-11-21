import re
import sys
import unicodedata

# Ensure UTF-8 output
sys.stdout.reconfigure(encoding='utf-8')

"""
Tanglish Spelling Scheme Notes:
--------------------------------
1. Vowels:
   - Short vowels: a, i, u, e, o
   - Long vowels: aa, ee (or i when syllable-final), oo (or u), e, o, ai, au
2. Consonants:
   - k, ng, ch/s, nj, t, n, th/dh, p/b/v, m, y, r, l, zh, s, sh, j, h
   - Intervocalic/post-nasal voicing: th -> dh (e.g. adhu, idhu, indha), p -> b/v
   - Hard retroflex/alveolar: tt, tr
   - Soft lateral retroflex: zh maintained consistently for ழ
3. Entities & Loanwords:
   - English words occurring in Tamil script are rendered in standard English orthography (e.g. CCTV, Facebook, Google Lens).
   - Entities have a canonical capitalization and single unified spelling (e.g. Delhi, Tamil, Vijay).
"""

# Independent vowels
VOWELS = {
    '\u0B85': 'a',   # அ
    '\u0B86': 'aa',  # ஆ
    '\u0B87': 'i',   # இ
    '\u0B88': 'ee',  # ஈ
    '\u0B89': 'u',   # உ
    '\u0B8A': 'oo',  # ஊ
    '\u0B8E': 'e',   # எ
    '\u0B8F': 'e',   # ஏ
    '\u0B90': 'ai',  # ஐ
    '\u0B92': 'o',   # ஒ
    '\u0B93': 'o',   # ஓ
    '\u0B94': 'au'   # ஔ
}

# Dependent vowel signs
VOWEL_SIGNS = {
    '\u0BBE': 'aa',  # ா
    '\u0BBF': 'i',   # ி
    '\u0BC0': 'ee',  # ீ
    '\u0BC1': 'u',   # ு
    '\u0BC2': 'oo',  # ூ
    '\u0BC6': 'e',   # ெ
    '\u0BC7': 'e',   # ே
    '\u0BC8': 'ai',  # ை
    '\u0BCA': 'o',   # ொ
    '\u0BCB': 'o',   # ோ
    '\u0BCC': 'au'   # ௌ
}

VIRAMA = '\u0BCD'  # ்
AYTHAM = '\u0B83'  # ஃ

# Consonants
CONSONANTS = {
    '\u0B95': 'k',   # க
    '\u0B99': 'ng',  # ங
    '\u0B9A': 'ch',  # ச
    '\u0B9E': 'nj',  # ஞ
    '\u0B9F': 't',   # ட
    '\u0BA3': 'n',   # ண
    '\u0BA4': 'th',  # த
    '\u0BA8': 'n',   # ந
    '\u0BA9': 'n',   # ன
    '\u0BAA': 'p',   # ப
    '\u0BAE': 'm',   # ம
    '\u0BAF': 'y',   # ய
    '\u0BB0': 'r',   # ர
    '\u0BB1': 'r',   # ற
    '\u0BB2': 'l',   # ல
    '\u0BB3': 'l',   # ள
    '\u0BB4': 'zh',  # ழ
    '\u0BB5': 'v',   # வ
    '\u0BB6': 's',   # ஶ
    '\u0BB7': 'sh',  # ஷ
    '\u0BB8': 's',   # ஸ
    '\u0BB9': 'h',   # ஹ
    '\u0B9C': 'j',   # ஜ
}

# Comprehensive dictionary of political parties, places, media, English loanwords, and common entities
ENTITY_MAP = {
    # Acronyms & Political parties
    'பாஜக': 'BJP',
    'பா.ஜ.க': 'BJP',
    'பா.ஜ.க.': 'BJP',
    'பாஜகவின்': 'BJP-yin',
    'பாஜகவுடன்': 'BJP-yudan',
    'பாஜகவில்': 'BJP-yil',
    'பாஜகவிற்கு': 'BJP-virku',
    'பாஜகவுக்கு': 'BJP-vukku',
    'பா.ஜ.க-வுடன்': 'BJP-yudan',
    'பா.ஜ.க-வால்': 'BJP-yaal',
    'பா.ஜ.க-வுக்கு': 'BJP-vukku',
    'திமுக': 'DMK',
    'தி.மு.க': 'DMK',
    'தி.மு.க.': 'DMK',
    'திமுகவின்': 'DMK-vin',
    'திமுகவில்': 'DMK-vil',
    'திமுகவுடன்': 'DMK-yudan',
    'திமுகவுக்கு': 'DMK-vukku',
    'திமுகவை': 'DMK-vai',
    'திமுகவினர்': 'DMK-vinar',
    'தி.மு.க-வில்': 'DMK-vil',
    'தி.மு.க-வுடன்': 'DMK-yudan',
    'தி.மு.க-வை': 'DMK-vai',
    'அதிமுக': 'ADMK',
    'அ.தி.மு.க': 'ADMK',
    'அ.தி.மு.க.': 'ADMK',
    'அதிமுகவின்': 'ADMK-vin',
    'அதிமுகவில்': 'ADMK-vil',
    'அதிமுகவுடன்': 'ADMK-yudan',
    'அதிமுகவுக்கு': 'ADMK-vukku',
    'அதிமுகவை': 'ADMK-vai',
    'அதிமுகவினர்': 'ADMK-vinar',
    'அதிமுக-வில்': 'ADMK-vil',
    'அ.தி.மு.க-வில்': 'ADMK-vil',
    'அ.தி.மு.க-வுடன்': 'ADMK-yudan',
    'அ.தி.மு.க-வை': 'ADMK-vai',
    'அ.தி.மு.க-வைச்': 'ADMK-vaich',
    'அ.தி.மு.க-வின்': 'ADMK-vin',
    'காங்கிரஸ்': 'Congress',
    'காங்கிரஸார்': 'Congress-ar',
    'காங்கிரஸின்': 'Congress-in',
    'காங்கிரஸில்': 'Congress-il',
    'காங்கிரசுக்கு': 'Congress-ukku',
    'காங்கிரஸை': 'Congress-ai',
    'தமாகா': 'TMC',
    'பாமக': 'PMK',
    'விசிக': 'VCK',
    'தவெக': 'TVK',
    'த.வெ.க': 'TVK',
    'டிவிகே': 'TVK',
    'நாம் தமிழர் கட்சி': 'Naam Tamilar Katchi',
    'நாதக': 'NTK',
    'தேஜகூட்டணி': 'NDA koottani',
    'தேசிய ஜனநாயக கூட்டணி': 'Dhesiya Jananaayaga Koottani (NDA)',
    'தேசிய ஜனநாயகக் கூட்டணி': 'Dhesiya Jananaayaga Koottani (NDA)',
    'சிஜேபி': 'CJP',
    'சி.ஜே.பி': 'CJP',

    # Technology, Social Media & Digital Loanwords
    'பேஸ்புக்': 'Facebook',
    'பேஸ்புக்கில்': 'Facebook-il',
    'பேஸ்புக்-ல்': 'Facebook-il',
    'ஃபேஸ்புக்': 'Facebook',
    'ஃபேஸ்புக்கில்': 'Facebook-il',
    'ஃபேஸ்புக்-ல்': 'Facebook-il',
    'வாட்ஸ்அப்': 'WhatsApp',
    'வாட்ஸ்அப்பில்': 'WhatsApp-il',
    'வாட்ஸ்அப்-ல்': 'WhatsApp-il',
    'வாட்ஸ்-அப்': 'WhatsApp',
    'ட்விட்டர்': 'Twitter',
    'ட்விட்டரில்': 'Twitter-il',
    'எக்ஸ்': 'X',
    'யூடியூப்': 'YouTube',
    'யூடியூபில்': 'YouTube-il',
    'இன்ஸ்டாகிராம்': 'Instagram',
    'இன்ஸ்டாகிராமில்': 'Instagram-il',
    'இன்ஸ்டா': 'Insta',
    'இன்ஸ்டாவில்': 'Insta-vil',
    'டிக்டாக்': 'TikTok',
    'டிக்டாக்கில்': 'TikTok-il',
    'கூகுள்': 'Google',
    'கூகுள் லென்ஸ்': 'Google Lens',
    'கூகுளில்': 'Google-il',
    'லென்ஸ்': 'Lens',
    'வீடியோ': 'video',
    'வீடியோவில்': 'video-vil',
    'வீடியோவை': 'video-vai',
    'வீடியோக்கள்': 'video-kkal',
    'வீடியோவின்': 'video-vin',
    'வீடியோவுக்கு': 'video-vukku',
    'வீடியோவாக': 'video-vaaga',
    'வீடியோவாகும்': 'video-vaagum',
    'வீடியோவா': 'video-vaa',
    'போட்டோ': 'photo',
    'போட்டோவை': 'photo-vai',
    'போட்டோக்கள்': 'photo-kkal',
    'போட்டோவில்': 'photo-vil',
    'புகைப்படம்': 'pugaippadam',
    'நியூஸ்': 'news',
    'நியூஸ் கார்டு': 'news card',
    'நியூஸ் கார்டை': 'news card-ai',
    'நியூஸ் கார்டில்': 'news card-il',
    'நியூஸ்கார்டு': 'news card',
    'கார்டு': 'card',
    'கார்டை': 'card-ai',
    'கார்டில்': 'card-il',
    'கார்டுகள்': 'card-gal',
    'கார்டும்': 'card-um',
    'ஹார்ட் டிஸ்க்': 'Hard disk',
    'ஹார்ட்': 'hard',
    'டிஸ்க்': 'disk',
    'ஸ்கிரிப்ட்': 'script',
    'ஸ்கிரிப்டட்': 'scripted',
    'சிசிடிவி': 'CCTV',
    'சிசிடிவியில்': 'CCTV-yil',
    'சிசிடிவியின்': 'CCTV-yin',
    'பேக்ட் செக்': 'Fact Check',
    'பேக்ட்': 'fact',
    'செக்': 'check',
    'ஃபேக்ட்': 'fact',
    'ஃபேக்ட்செக்': 'Fact Check',
    'ஃபேக்ட் செக்': 'Fact Check',
    'பீட்சா': 'pizza',
    'டெலிவரி': 'delivery',
    'விசா': 'visa',
    'விசாக்கள்': 'visas',
    'எச்-1பி': 'H-1B',
    'ஹெச்-1பி': 'H-1B',
    'கிரிக்கெட்': 'cricket',
    'மேட்ச்': 'match',
    'சிஎஸ்கே': 'CSK',
    'வைரல்': 'viral',
    'வைரலாக': 'viral-aaga',
    'வைரலாகி': 'viral-aagi',
    'வைரலாகும்': 'viral-aagum',
    'சோஷியல் மீடியா': 'social media',
    'ஷேர்': 'share',
    'கமெண்ட்': 'comment',
    'லைவ்': 'live',
    'போஸ்ட்': 'post',
    'இணையதளம்': 'inaiyathalam',
    'இணையதளத்தில்': 'inaiyathalathil',

    # Months
    'ஜனவரி': 'January',
    'பிப்ரவரி': 'February',
    'மார்ச்': 'March',
    'ஏப்ரல்': 'April',
    'ஜூன்': 'June',
    'ஜூலை': 'July',
    'ஆகஸ்ட்': 'August',
    'செப்டம்பர்': 'September',
    'செப்டம்பரில்': 'September-il',
    'அக்டோபர்': 'October',
    'அக்டோபரில்': 'October-il',
    'நவம்பர்': 'November',
    'நவம்பரில்': 'November-il',
    'டிசம்பர்': 'December',
    'டிசம்பரில்': 'December-il',

    # Law enforcement & Official
    'போலீஸ்': 'Police',
    'போலீசார்': 'Police',
    'போலீசாரால்': 'Police-araal',
    'போலீசாரிடம்': 'Police-aridam',
    'போலீசுக்கு': 'Police-ukku',
    'கலெக்டர்': 'Collector',
    'கலெக்டரிடம்': 'Collector-idam',
    'கலெக்டரின்': 'Collector-in',
    'சிபிஐ': 'CBI',
    'என்ஐஏ': 'NIA',
    'நீதிமன்றம்': 'Needhimandram',
    'உயர்நீதிமன்றம்': 'Uyarneedhimandram',
    'உச்சநீதிமன்றம்': 'Uchaneedhimandram',
    'ஐகோர்ட்': 'High Court',
    'சுப்ரீம் கோர்ட்': 'Supreme Court',
    'கோர்ட்': 'Court',
    'கோர்ட்டில்': 'Court-il',
    'எப்ஐஆர்': 'FIR',
    'டாஸ்மாக்': 'TASMAC',

    # Common English loanwords
    'பிளாக்': 'black',
    'பிளாக் டிக்கெட்': 'black ticket',
    'டிக்கெட்': 'ticket',
    'ரயில்': 'rail',
    'ரயில்வே': 'railway',
    'ரயில்வே ஸ்டேஷனில்': 'railway station-il',
    'ஸ்டேஷன்': 'station',
    'ஸ்டேஷனில்': 'station-il',
    'பஸ்': 'bus',
    'கார்': 'car',
    'ஆட்டோ': 'auto',
    'பைக்': 'bike',
    'ஸ்கூட்டர்': 'scooter',
    'டாக்டர்': 'doctor',
    'நர்ஸ்': 'nurse',
    'டிவி': 'TV',
    'டிவியில்': 'TV-yil',
    'சேனல்': 'channel',
    'மைக்': 'mic',
    'மீட்டிங்': 'meeting',
    'கிலோ': 'kilo',
    'மீட்டர்': 'meter',
    'கிமீ': 'km',
    'லிட்டர்': 'liter',
    'சதவீதம்': 'sadhaveedham',
    'கோடி': 'kodi',
    'லட்சம்': 'latcham',
    'ஆயிரம்': 'aayiram',
    'ரூபாய்': 'roobai',
    'ரூபாயாக': 'roobaayaaga',

    # Places & Countries (Strictly standardized)
    'இந்தியா': 'India',
    'இந்திய': 'India',
    'இந்தியாவில்': 'India-vil',
    'இந்தியாவுக்கு': 'India-vukku',
    'இந்தியாவின்': 'India-vin',
    'இந்தியாவுக்கான': 'India-vukkaana',
    'இந்தியாவுக்கும்': 'India-vukkum',
    'இந்தியாவிடம்': 'India-vidam',
    'இந்தியாவிலிருந்து': 'India-vilirundhu',
    'இந்தியாவுக்குள்': 'India-vukkul',
    'இந்தியாவுடன்': 'India-yudan',
    'இந்தியாவும்': 'India-vum',
    'இந்தியாவை': 'India-vai',
    'இந்தியாவைச்': 'India-vaich',
    'ரஷ்யா': 'Russia',
    'ரஷ்ய': 'Russia',
    'ரஷ்யாவின்': 'Russia-vin',
    'ரஷ்யாவில்': 'Russia-vil',
    'ரஷ்யாவுக்கு': 'Russia-vukku',
    'ரஷ்ய தூதர்': 'Russia thoodhar',
    'அமெரிக்கா': 'America',
    'அமெரிக்க': 'America',
    'அமெரிக்காவின்': 'America-vin',
    'அமெரிக்காவில்': 'America-vil',
    'அமெரிக்காவுக்கு': 'America-vukku',
    'அமெரிக்காவுக்கான': 'America-vukkaana',
    'அமெரிக்காவுக்கும்': 'America-vukkum',
    'அமெரிக்காவை': 'America-vai',
    'அமெரிக்காவைத்': 'America-vaith',
    'அமெரிக்காவைப்': 'America-vaip',
    'உக்ரைன்': 'Ukraine',
    'உக்ரைனின்': 'Ukraine-in',
    'சீனா': 'China',
    'சீன': 'China',
    'சீனாவின்': 'China-vin',
    'ஜப்பான்': 'Japan',
    'ஜப்பானில்': 'Japan-il',
    'பாகிஸ்தான்': 'Pakistan',
    'பாகிஸ்தானில்': 'Pakistan-il',
    'இஸ்ரேல்': 'Israel',
    'இஸ்ரேலில்': 'Israel-il',
    'பாலஸ்தீனம்': 'Palestine',
    'ஈரான்': 'Iran',
    'ஈரானில்': 'Iran-il',
    'ஈரானின்': 'Iran-in',
    'ஈராக்': 'Iraq',
    'லக்னோ': 'Lucknow',
    'லக்னோவில்': 'Lucknow-vil',
    'டெல்லி': 'Delhi',
    'டெல்லியில்': 'Delhi-yil',
    'டெல்லிக்கு': 'Delhi-kku',
    'டெல்லியிலிருந்து': 'Delhi-yilirundhu',
    'தில்லி': 'Delhi',
    'தில்லியில்': 'Delhi-yil',
    'தில்லிக்கு': 'Delhi-kku',
    'தில்லியிலிருந்து': 'Delhi-yilirundhu',
    'டில்லி': 'Delhi',
    'டில்லியில்': 'Delhi-yil',
    'டில்லிக்கு': 'Delhi-kku',
    'டில்லியிலிருந்து': 'Delhi-yilirundhu',
    'புதுடில்லி': 'Delhi',
    'புதுடெல்லி': 'Delhi',
    'புதுதில்லி': 'Delhi',
    'புதுதில்லியில்': 'Delhi-yil',
    'புதுடில்லியில்': 'Delhi-yil',
    'புதுடெல்லியில்': 'Delhi-yil',
    'சென்னை': 'Chennai',
    'சென்னையில்': 'Chennai-yil',
    'சென்னையின்': 'Chennaiyin',
    'தமிழ்': 'Tamil',
    'தமிழில்': 'Tamil-il',
    'தமிழுக்கு': 'Tamil-ukku',
    'தமிழ்நாடு': 'Tamil Nadu',
    'தமிழ்நாட்டின்': 'Tamil Nadu-vin',
    'தமிழ்நாட்டில்': 'Tamil Nadu-vil',
    'தமிழ்நாட்டிற்கும்': 'Tamil Nadu-virgum',
    'தமிழ்நாட்டு': 'Tamil Nadu',
    'தமிழக': 'Tamil Nadu',
    'தமிழகத்தில்': 'Tamil Nadu-vil',
    'தமிழக அரசு': 'Tamil Nadu arasu',
    'கோயம்புத்தூர்': 'Coimbatore',
    'கோவை': 'Kovai',
    'மதுரை': 'Madurai',
    'மதுரையில்': 'Maduraiyil',
    'திருச்சி': 'Trichy',
    'சேலம்': 'Salem',
    'திருப்பூர்': 'Tiruppur',
    'திருநெல்வேலி': 'Tirunelveli',
    'நெல்லை': 'Nellai',
    'கிருஷ்ணகிரி': 'Krishnagiri',
    'கிருஷ்ணகிரியில்': 'Krishnagiriyil',
    'மேட்டுப்பாளையம்': 'Mettupalayam',
    'மேட்டுப்பாளையத்தில்': 'Mettupalayathil',
    'ஈரோடு': 'Erode',
    'தஞ்சாவூர்': 'Thanjavur',
    'செங்கல்பட்டு': 'Chengalpattu',
    'திருவள்ளூர்': 'Tiruvallur',
    'கும்மிடிப்பூண்டி': 'Gummidipoondi',
    'தூத்துக்குடி': 'Thoothukudi',
    'தூத்துக்குடியில்': 'Thoothukudiyil',
    'ஆத்தூர்': 'Aathoor',
    'கடையநல்லூர்': 'Kadayanallur',

    # Leaders & Named Entities (Strict Capitalization & Unified Spelling)
    'ஸ்டாலின்': 'Stalin',
    'மு.க.ஸ்டாலின்': 'M.K. Stalin',
    'மு.க.ஸ்டாலினின்': 'M.K. Stalin-in',
    'மோடி': 'Modi',
    'மோடியின்': 'Modi-yin',
    'மோடிக்கு': 'Modi-kku',
    'நரேந்திர மோடி': 'Narendra Modi',
    'விஜய்': 'Vijay',
    'விஜயின்': 'Vijay-in',
    'விஜய்க்கு': 'Vijay-kku',
    'விஜய்யின்': 'Vijay-yin',
    'விஜயை': 'Vijay-ai',
    'ஞானேஷ்': 'Gyanesh',
    'ஞானேஷ் குமார்': 'Gyanesh Kumar',
    'ஞானேஷ்குமார்': 'Gyanesh Kumar',
    'ஞானேஷ்குமாரால்': 'Gyanesh Kumar-aal',
    'சுக்பீர்': 'Sukhbir',
    'சுக்பீர் சிங் சந்து': 'Sukhbir Singh Sandhu',
    'சுக்பீர் சிங்': 'Sukhbir Singh',
    'விவேக் ஜோஷி': 'Vivek Joshi',
    'அபிஜீத்': 'Abhijit',
    'அபிஜீத் தீப்கே': 'Abhijit Deepke',
    'தீப்கே': 'Deepke',
    'தமிழ்செல்வன்': 'Tamilselvan',
    'தமிழ்ச்செல்வன்': 'Tamilselvan',
    'தமிழ்செல்வனின்': 'Tamilselvan-in',
    'அன்பில் மகேஸ்': 'Anbil Mahesh',
    'அன்பில் மகேஸிடம்': 'Anbil Mahesh-idam',
    'எடப்பாடி': 'Edappadi',
    'பழனிசாமி': 'Palaniswami',
    'அண்ணாமலை': 'Annamalai',
    'அண்ணாமலையின்': 'Annamalai-yin',
    'அண்ணாமலையுடன்': 'Annamalaiyudan',
    'சீமான்': 'Seeman',
    'ராகுல் காந்தி': 'Rahul Gandhi',
    'அமித் ஷா': 'Amit Shah',
    'அம்பேத்கர்': 'Ambedkar',
    'ட்ரம்ப்': 'Trump',
    'டிரம்ப்': 'Trump',
    'புதின்': 'Putin',
    'பைடன்': 'Biden',
    'ஜெயக்குமார்': 'Jayakumar',
    'ஜெயகுமார்': 'Jayakumar',
    'துரைமுருகன்': 'Duraimurugan',
    'உதயநிதி': 'Udhayanidhi',
    'அப்பாவு': 'Appavu',
    'ஜெஃப்ரி': 'Jeffrey',
    'எப்ஸ்டீன்': 'Epstein',
    'கமேனி': 'Khamenei',
    'அயதுல்லா': 'Ayatollah',
    'பாரதிதாசன்': 'Bharathidhasan',
    'கார்த்திகேயன்': 'Karthikeyan',

    # Media channels
    'பாலிமர்': 'Polimer',
    'பாலிமர் செய்திகள்': 'Polimer Seithigal',
    'சன் நியூஸ்': 'Sun News',
    'சன் டிவி': 'Sun TV',
    'புதிய தலைமுறை': 'Puthiya Thalaimurai',
    'தந்தி டிவி': 'Thanthi TV',
    'நியூஸ் 18': 'News 18',
    'நியூஸ்7': 'News 7',
    'பிடிஐ': 'PTI',
    'ஏஎன்ஐ': 'ANI',
    'சிஎன்என்': 'CNN',
    'பிஐபி': 'PIB',
    'பிஐபி ஃபேக்ட் செக்': 'PIB Fact Check',
    'எஸ்ஐஆர்': 'SIR',
    'எஸ்.ஐ.ஆர்': 'SIR',

    # Single components of multi-word names & loanwords
    'குமார்': 'Kumar',
    'சிங்': 'Singh',
    'சந்து': 'Sandhu',
    'விவேக்': 'Vivek',
    'ஜோஷி': 'Joshi',
    'காந்தி': 'Gandhi',
    'ஷா': 'Shah',
    'திரிஷா': 'Trisha',
    'மம்தா': 'Mamata',
    'சாஸ்திரி': 'Shastri',
    'பியூஸ்': 'Piyush',
    'கோயல்': 'Goyal',
    'அஜித்': 'Ajith',
    'அஜித்குமார்': 'Ajith Kumar',
    'விஷால்': 'Vishal',
    'ராஷ்மிகா': 'Rashmika',
    'பானர்ஜி': 'Banerjee',
    'ரஜினிகாந்த்': 'Rajinikanth',
    'ராஜ்பார்': 'Rajbhar',
    'ஸ்ட்ரீட்': 'street',
    'வியூ': 'view',
    'வியூவ்': 'view',
    'ஸ்ட்ரீட் வியூ': 'street view',
    'மேப்': 'map',
    'மேப்பில்': 'map-il',
    'மேப்-ல்': 'map-il',
    'கூகுள் மேப்': 'Google Map',
    'கூகுள் மேப்பில்': 'Google Map-il',
    'கூகுள் மேப்-ல்': 'Google Map-il',
    'ஃபேக்ட் கிரஸண்டோ': 'Fact Crescendo',
    'பேக்ட் கிரஸண்டோ': 'Fact Crescendo',
    'ஃபேக்ட் கிரசண்டோ': 'Fact Crescendo',
    'பேக்ட் கிரசண்டோ': 'Fact Crescendo',
    'கிரஸண்டோ': 'Crescendo',
    'கிரசண்டோ': 'Crescendo',
    'சிஎஸ்கே கிரிக்கெட் மேட்ச்': 'CSK cricket match',
    'சி.எஸ்.கே': 'CSK',
    'எச்-1பி': 'H-1B',
    'எச் - 1 பி': 'H-1B',
    'ஹெச்-1பி': 'H-1B',
    'எஸ்-1பி': 'H-1B',
    'எச்-1பி விசா': 'H-1B visa',
    'எச் - 1 பி விசா': 'H-1B visa',
    'ஹெச்-1பி விசா': 'H-1B visa',
    'எஸ்-1பி விசா': 'H-1B visa',
    'விசாக்கள்': 'visas',
    'விசா': 'visa',
    'ஸ்கிரீன்ஷாட்': 'screenshot',
    'ஸ்கிரீன்ஷாட்கள்': 'screenshots',
    'ஸ்கிரீன்ஷாட்களில்': 'screenshots-il',
    'ஸ்கிரீன்ஷாட்களை': 'screenshots-ai',
    'மணிகண்டன்': 'Manikandan',
    'யோகலட்சுமி': 'Yogalakshmi',
    'கரூரில்': 'Karur-il',
    'கரூர்': 'Karur',
}

def normalize_tamil(text):
    if not text or not isinstance(text, str):
        return text
    
    # Unicode NFC normalization
    text = unicodedata.normalize('NFC', text)
    
    # Normalize non-breaking spaces (\xa0) and non-standard whitespace
    text = text.replace('\xa0', ' ')
    text = re.sub(r'[\u200B-\u200D\uFEFF]', '', text) # Zero-width spaces
    
    # Normalize em-dashes and hyphens for consistency (e.g. "India - Russia")
    text = re.sub(r'\s*[\u2013\u2014]\s*|\s*--\s*', ' - ', text)
    
    # Fix inverted left-side vowel signs (ெ, ே, ை typed before consonant)
    text = re.sub(r'(\u0BCD)([\u0BC6\u0BC7\u0BC8])([\u0B95-\u0BB9])', r'\1\3\2', text)
    text = re.sub(r'([\u0BBE-\u0BCC])([\u0BC6\u0BC7\u0BC8])([\u0B95-\u0BB9])', r'\1\3\2', text)
    text = re.sub(r'(?<![\u0B95-\u0BB9])([\u0BC6\u0BC7\u0BC8])([\u0B95-\u0BB9])', r'\2\1', text)
    # Fix legacy typesetting decomposed glyphs
    text = re.sub(r'\u0BC1\u0BBE', '\u0BC2', text) # ு + ா -> ூ
    text = re.sub(r'\u0BC6\u0BBE', '\u0BCA', text) # ெ + ா -> ொ
    text = re.sub(r'\u0BC7\u0BBE', '\u0BCB', text) # ே + ா -> ோ
    text = re.sub(r'\u0BC6\u0BD7', '\u0BCC', text) # ெ + ௗ -> ௌ
    # Legacy font ligatures
    text = re.sub(r'வௌி', 'வெளி', text)
    text = re.sub(r'ாி', 'ரி', text)
    text = re.sub(r'ா்', 'ர்', text)
    
    # CRITICAL: Fix isolated prefix that caused 'vavijay' regression:
    # (?<![\u0B95-\u0BB9])ிஜய் -> விஜய்
    text = re.sub(r'(?<![\u0B95-\u0BB9])ிஜய்', 'விஜய்', text)
    
    # Strip isolated / misplaced vowel signs or viramas
    text = re.sub(r'(?<![\u0B95-\u0BB9])\u0BBF', 'i', text)
    text = re.sub(r'(?<![\u0B95-\u0BB9])\u0BC0', 'ee', text)
    text = re.sub(r'(?<![\u0B95-\u0BB9])\u0BBE', 'aa', text)
    text = re.sub(r'(?<![\u0B95-\u0BB9])\u0BC8', 'ai', text)
    text = re.sub(r'(?<![\u0B95-\u0BB9])\u0BC7', 'e', text)
    text = re.sub(r'(?<![\u0B95-\u0BB9])\u0BC6', 'e', text)
    text = re.sub(r'(?<![\u0B95-\u0BB9])\u0BCD', '', text)
    return text


def parse_syllables(tamil_word):
    syllables = []
    i = 0
    chars = list(tamil_word)
    n = len(chars)

    while i < n:
        c = chars[i]

        # Check Aytham (ஃ) followed by ப -> 'f'
        if c == AYTHAM:
            if i + 1 < n and chars[i+1] == '\u0BAA': # ஃ + ப
                syllables.append({'type': 'aytham_f', 'consonant': None, 'vowel': None, 'raw': c})
                i += 1
                continue
            else:
                syllables.append({'type': 'aytham', 'consonant': None, 'vowel': None, 'raw': c})
                i += 1
                continue

        if c in VOWELS:
            syllables.append({
                'type': 'vowel',
                'consonant': None,
                'vowel': VOWELS[c],
                'raw': c
            })
            i += 1
            continue

        if c in CONSONANTS:
            cons_char = c
            if i + 1 < n and chars[i+1] == VIRAMA:
                syllables.append({
                    'type': 'pure_consonant',
                    'consonant': cons_char,
                    'vowel': None,
                    'raw': cons_char + VIRAMA
                })
                i += 2
            elif i + 1 < n and chars[i+1] in VOWEL_SIGNS:
                syllables.append({
                    'type': 'syllable',
                    'consonant': cons_char,
                    'vowel': VOWEL_SIGNS[chars[i+1]],
                    'raw': cons_char + chars[i+1]
                })
                i += 2
            else:
                # Inherent vowel 'a'
                syllables.append({
                    'type': 'syllable',
                    'consonant': cons_char,
                    'vowel': 'a',
                    'raw': cons_char
                })
                i += 1
            continue

        syllables.append({
            'type': 'other',
            'consonant': None,
            'vowel': None,
            'raw': c
        })
        i += 1

    return syllables


def transliterate_word(word):
    # Match enclosing quotes, brackets, punctuation
    match = re.match(r'^([\"“”\‘\’\'\(\[\{]*)([\u0B80-\u0BFF\w\.\-]+)([\"“”\‘\’\'\)\?\,\!\:\;\.\%]*)$', word)
    prefix, core, suffix = '', word, ''
    if match:
        prefix, core, suffix = match.groups()

    if core in ENTITY_MAP:
        return prefix + ENTITY_MAP[core] + suffix

    core_no_dots = core.replace('.', '')
    if core_no_dots in ENTITY_MAP:
        return prefix + ENTITY_MAP[core_no_dots] + suffix

    # Rule 3e: Standardise suffixes: English/Latin stem + Tamil suffix (e.g. Reuters-in, Party-yin, MP-aa, threads.com-il)
    hybrid_match = re.match(r'^([A-Za-z0-9\.\_\@\:\/\-]+?)-?([\u0B80-\u0BFF]+)$', core)
    if hybrid_match:
        stem, tam_suffix = hybrid_match.groups()
        if tam_suffix in ('ன்', 'வின்', 'இன்'):
            trans_suf = 'in' if tam_suffix == 'ன்' else ('vin' if tam_suffix == 'வின்' else 'in')
        elif tam_suffix in ('ல்', 'வில்', 'இல்'):
            trans_suf = 'il' if tam_suffix == 'ல்' else ('vil' if tam_suffix == 'வில்' else 'il')
        elif tam_suffix in ('க்கு', 'வுக்கு', 'ற்கு', 'விற்கு'):
            trans_suf = 'kku' if tam_suffix == 'க்கு' else ('vukku' if tam_suffix == 'வுக்கு' else 'ku')
        elif tam_suffix in ('ஆ', 'வா'):
            trans_suf = 'aa' if tam_suffix == 'ஆ' else 'vaa'
        elif tam_suffix in ('யும்', 'வும்', 'உம்'):
            trans_suf = 'yum' if tam_suffix == 'யும்' else ('vum' if tam_suffix == 'வும்' else 'um')
        elif tam_suffix in ('யின்', 'யை', 'யில்', 'யுடன்'):
            suf_map = {'யின்': 'yin', 'யை': 'yai', 'யில்': 'yil', 'யுடன்': 'yudan'}
            trans_suf = suf_map[tam_suffix]
        else:
            trans_suf = transliterate_word(tam_suffix).lstrip('-')
        return f"{prefix}{stem}-{trans_suf}{suffix}"

    syllables = parse_syllables(core)
    num_syl = len(syllables)
    result = []

    for idx, syl in enumerate(syllables):
        stype = syl['type']
        if stype == 'other':
            result.append(syl['raw'])
            continue
        if stype == 'aytham_f':
            continue
        if stype == 'aytham':
            result.append('k')
            continue
        if stype == 'vowel':
            result.append(syl['vowel'])
            continue

        cons = syl['consonant']
        vowel = syl['vowel'] or ''
        prev_syl = syllables[idx-1] if idx > 0 else None
        next_syl = syllables[idx+1] if idx + 1 < num_syl else None

        cons_str = CONSONANTS.get(cons, '')

        if stype == 'pure_consonant':
            # Pure consonant (pulli)
            if cons == '\u0BB1': # ற்
                if next_syl and next_syl.get('consonant') == '\u0BB1':
                    cons_str = ''
                elif next_syl and next_syl.get('consonant') in ('\u0B95', '\u0BAA'):
                    cons_str = 'r'
                else:
                    cons_str = 't'
            elif cons == '\u0BA4': # த்
                if next_syl and next_syl.get('consonant') == '\u0BA4':
                    cons_str = ''
                else:
                    cons_str = 'th'
            elif cons == '\u0B9A': # ச்
                if next_syl and next_syl.get('consonant') == '\u0B9A':
                    cons_str = ''
                else:
                    cons_str = 's'
            elif cons == '\u0B9F': # ட்
                if next_syl and next_syl.get('consonant') == '\u0B9F':
                    cons_str = 't'
                elif next_syl and next_syl.get('consonant') == '\u0B9A':
                    cons_str = 't'
                else:
                    cons_str = 't'
            elif cons == '\u0B95': # க்
                cons_str = 'k'
            elif cons == '\u0BAA': # ப்
                cons_str = 'p'
            elif cons == '\u0BAE': # ம்
                cons_str = 'm'
            elif cons == '\u0BA8': # ந்
                cons_str = 'n'
            elif cons == '\u0BA9': # ன்
                cons_str = 'n'
            elif cons == '\u0BA3': # ண்
                cons_str = 'n'
            elif cons == '\u0B99': # ங்
                cons_str = 'ng'
            elif cons == '\u0B9E': # ஞ்
                cons_str = 'nj'
            elif cons == '\u0BB4': # ழ்
                cons_str = 'zh'
            elif cons == '\u0BB3': # ள்
                cons_str = 'l'
            elif cons == '\u0BB2': # ல்
                cons_str = 'l'
            elif cons == '\u0BB0': # ர்
                cons_str = 'r'
            elif cons == '\u0BB5': # வ்
                cons_str = 'v'
            elif cons == '\u0BB7': # ஷ்
                cons_str = 'sh'
            elif cons == '\u0BB8': # ஸ்
                cons_str = 's'
            elif cons == '\u0B9C': # ஜ்
                cons_str = 'j'
            elif cons == '\u0BB9': # ஹ்
                cons_str = 'h'

            result.append(cons_str)

        else:
            # Syllable with vowel
            is_initial = (idx == 0 or prev_syl['type'] == 'other')
            prev_is_vowel = (prev_syl is not None and prev_syl['type'] in ('vowel', 'syllable'))
            prev_is_pure = (prev_syl is not None and prev_syl['type'] == 'pure_consonant')
            prev_cons = prev_syl.get('consonant') if prev_is_pure else None
            prev_is_aytham_f = (prev_syl is not None and prev_syl['type'] == 'aytham_f')

            if cons == '\u0B95': # க
                if is_initial:
                    cons_str = 'k'
                elif prev_cons == '\u0B95': # க்க -> kk
                    cons_str = 'k'
                elif prev_cons == '\u0B99': # ங்க -> ng
                    cons_str = ''
                elif prev_cons == '\u0BB1': # ற்க -> rk
                    cons_str = 'k'
                elif prev_cons == '\u0B9F': # ட்க -> tk
                    cons_str = 'k'
                elif prev_is_vowel:
                    cons_str = 'g'
                else:
                    cons_str = 'g'

            elif cons == '\u0B9A': # ச
                if is_initial:
                    cons_str = 's'
                elif prev_cons == '\u0B9A': # ச்ச -> ch
                    cons_str = 'ch'
                elif prev_cons == '\u0B9E': # ஞ்ச -> nj
                    cons_str = 'j'
                elif prev_cons == '\u0B9F': # ட்ச -> tch
                    cons_str = 'ch'
                elif prev_is_vowel:
                    cons_str = 's'
                else:
                    cons_str = 's'

            elif cons == '\u0B9F': # ட
                if is_initial:
                    cons_str = 't'
                elif prev_cons == '\u0B9F': # ட்ட -> tt
                    cons_str = 't'
                elif prev_cons == '\u0BA3': # ண்ட -> nd
                    cons_str = 'd'
                elif prev_is_vowel:
                    cons_str = 'd'
                else:
                    cons_str = 'd'

            elif cons == '\u0BA4': # த
                if is_initial:
                    cons_str = 'th'
                elif prev_cons == '\u0BA4': # த்த -> thth
                    cons_str = 'th'
                elif prev_cons == '\u0BA8': # ந்த -> ndh
                    cons_str = 'dh'
                elif prev_cons == '\u0BAF': # ய்த -> ith
                    cons_str = 'th'
                elif prev_is_vowel:
                    cons_str = 'dh'
                else:
                    cons_str = 'dh'

            elif cons == '\u0BAA': # ப
                if prev_is_aytham_f: # ஃ + ப -> f
                    cons_str = 'f'
                elif is_initial:
                    cons_str = 'p'
                elif prev_cons == '\u0BAA': # ப்ப -> pp
                    cons_str = 'p'
                elif prev_cons == '\u0BAE': # ம்ப -> mb
                    cons_str = 'b'
                elif prev_cons == '\u0BB1': # ற்ப -> rp
                    cons_str = 'p'
                elif prev_is_vowel:
                    cons_str = 'v'
                else:
                    cons_str = 'b'

            elif cons == '\u0BB1': # ற
                if is_initial:
                    cons_str = 'r'
                elif prev_cons == '\u0BB1': # ற்ற -> tra / tru / tri
                    if vowel in ('a', 'aa'):
                        cons_str = 'tra'
                        vowel = ''
                    elif vowel in ('u', 'oo'):
                        cons_str = 'tru'
                        vowel = ''
                    elif vowel in ('i', 'ee'):
                        cons_str = 'tri'
                        vowel = ''
                    else:
                        cons_str = 'tr'
                elif prev_cons == '\u0BA9': # ன்ற -> ndra
                    cons_str = 'dr'
                elif prev_is_vowel:
                    cons_str = 'r'
                else:
                    cons_str = 'r'

            elif cons == '\u0BA8': # ந
                cons_str = 'n'
            elif cons == '\u0BA9': # ன
                cons_str = 'n'
            elif cons == '\u0BA3': # ண
                cons_str = 'n'
            elif cons == '\u0BAF': # ய
                cons_str = 'y'
            elif cons == '\u0BB0': # ர
                cons_str = 'r'
            elif cons == '\u0BB2': # ல
                cons_str = 'l'
            elif cons == '\u0BB3': # ள
                cons_str = 'l'
            elif cons == '\u0BB4': # ழ
                cons_str = 'zh'
            elif cons == '\u0BB5': # வ
                cons_str = 'v'
            elif cons == '\u0B99': # ங
                cons_str = 'ng'
            elif cons == '\u0B9E': # ஞ
                cons_str = 'nj'
            elif cons == '\u0B9C': # ஜ
                cons_str = 'j'
            elif cons == '\u0BB7': # ஷ
                cons_str = 'sh'
            elif cons == '\u0BB8': # ஸ
                cons_str = 's'
            elif cons == '\u0BB9': # ஹ
                cons_str = 'h'

            result.append(cons_str + vowel)

    raw_res = "".join(result)

    # Post-cleanups for standard readable Tanglish conventions
    raw_res = re.sub(r'seydh', 'seith', raw_res)
    raw_res = re.sub(r'seydha', 'seidha', raw_res)
    raw_res = re.sub(r'vittru', 'vitru', raw_res)
    raw_res = re.sub(r'gittaye', 'kitteye', raw_res)
    raw_res = re.sub(r'gitta', 'kitta', raw_res)

    return prefix + raw_res + suffix


def normalize_quotes_and_typography(text):
    """Normalize quotes to canonical curved quotes and clean duplicate quotes and spaces."""
    if not text:
        return text

    # Remove non-breaking spaces
    text = text.replace('\xa0', ' ')

    # Fix messy punctuation combinations like .,'' or ,.'' or .,
    text = re.sub(r'\.\s*,\s*[\'\"‘’’“”]*', '.', text)
    text = re.sub(r',\s*\.\s*[\'\"‘’’“”]*', ',', text)
    text = re.sub(r'([.,])\s*[\'\"‘’’“”]{2,}', r'\1', text)

    # Convert doubled quotes to proper open/close quotes
    # Opening doubled quotes (preceding a word or at start)
    text = re.sub(r'(^|[\s\(\[\{])([‘\'’]{2,}|[“"”]{2,})', r'\1“', text)
    # Closing doubled quotes (following a word or punctuation)
    text = re.sub(r'([^\s])([‘\'’]{2,}|[“"”]{2,})', r'\1”', text)
    # Any residual doubled quotes
    text = re.sub(r'[‘\'’]{2,}', '“', text)
    text = re.sub(r'[“"”]{2,}', '”', text)

    # Collapse multiple spaces (preserving single space and newlines)
    text = re.sub(r'[^\S\r\n]{2,}', ' ', text)

    return text


def transliterate_text(text):
    if not text or not isinstance(text, str):
        return text

    # Step 1: Pre-normalization of Tamil text, hyphens, and whitespace
    text = normalize_tamil(text)

    # Multi-word entity and phrase replacements (sorted by descending length)
    multi_word_entities = sorted(
        [(k, v) for k, v in ENTITY_MAP.items() if ' ' in k],
        key=lambda x: len(x[0]),
        reverse=True
    )
    for k, v in multi_word_entities:
        pattern = r'(?<![\u0B80-\u0BFF])' + re.escape(k) + r'(?![\u0B80-\u0BFF])'
        text = re.sub(pattern, v, text)

    lines = text.split('\n')
    trans_lines = []

    for line in lines:
        if not line.strip():
            trans_lines.append(line)
            continue

        tokens = line.split(' ')
        trans_tokens = []
        for token in tokens:
            if not token:
                trans_tokens.append('')
                continue

            if re.search(r'[\u0B80-\u0BFF]', token):
                trans_tokens.append(transliterate_word(token))
            else:
                # Rule 3a: Preserve Latin-script words already in the source exactly (case included)
                trans_tokens.append(token)

        trans_lines.append(" ".join(trans_tokens))

    out_text = "\n".join(trans_lines)

    # Sentence boundary capitalization: handle initial quotes/brackets gracefully
    def cap_first_alpha(sentence_str):
        for idx, char in enumerate(sentence_str):
            if char.isalpha():
                return sentence_str[:idx] + char.upper() + sentence_str[idx+1:]
        return sentence_str

    sentences = re.split(r'([\.\?\!]\s+)', out_text)
    capped_sentences = [cap_first_alpha(s) if s else s for s in sentences]
    out_text = "".join(capped_sentences)

    # Step 3: Typography & Quote normalization
    out_text = normalize_quotes_and_typography(out_text)

    # Regression safeguard: "vavijay" must never occur
    out_text = re.sub(r'\bvavijay\b', 'Vijay', out_text, flags=re.IGNORECASE)
    out_text = re.sub(r'vavijay', 'Vijay', out_text, flags=re.IGNORECASE)

    return out_text


def test_regression_vavijay():
    """Rule 3d: Add a regression test: 'விஜய்' must produce 'Vijay', never 'vavijay'."""
    assert transliterate_text("விஜய்") == "Vijay", "Regression failed: விஜய் must produce Vijay"
    assert "vavijay" not in transliterate_text("விஜய்").lower(), "Regression failed: vavijay found"
    assert "vavijay" not in transliterate_text("அங்கு விஜய் வந்தார்").lower(), "Regression failed: vavijay in sentence"
    return True

# Run regression test on module import
test_regression_vavijay()

