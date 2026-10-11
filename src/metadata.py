"""Static clinical info shown next to each prediction (carried over from the Flask app)."""

DISEASE_METADATA = {
    "COVID-19": {
        "description": "A highly contagious viral infection caused by the SARS-CoV-2 virus. Symptoms range from mild respiratory issues to severe pneumonia.",
        "severity": "High",
        "precautions": [
            "Isolate yourself in a well-ventilated room away from others.",
            "Wear a high-filtering mask (e.g. N95) if contact is unavoidable.",
            "Monitor blood oxygen levels regularly using a pulse oximeter.",
            "Stay hydrated and take fever-reducing medication (e.g. Acetaminophen) if advised.",
            "Seek immediate emergency care if you experience difficulty breathing."
        ]
    },
    "Flu": {
        "description": "Influenza is a common respiratory infection caused by influenza viruses. It onset is typically sudden and can affect the entire body.",
        "severity": "Medium",
        "precautions": [
            "Get plenty of bed rest to allow your immune system to recover.",
            "Drink ample fluids (water, herbal teas, broths) to prevent dehydration.",
            "Take over-the-counter pain relievers or decongestants for symptom relief.",
            "Avoid contact with others to prevent spreading the virus.",
            "Consult a doctor within 48 hours for antiviral prescription if you are high-risk."
        ]
    },
    "Cold": {
        "description": "A mild viral infection of the upper respiratory tract. It is extremely common and usually resolves within a week.",
        "severity": "Low",
        "precautions": [
            "Rest as much as possible and stay warm.",
            "Stay hydrated by drinking warm fluids (tea, warm water).",
            "Use saline nasal drops or a humidifier to relieve nasal congestion.",
            "Gargle with warm salt water to soothe a sore throat.",
            "Consider vitamin C or zinc supplements to boost recovery."
        ]
    },
    "Allergy": {
        "description": "An immune system reaction to foreign substances (allergens) like pollen, dust mites, mold, or pet dander.",
        "severity": "Low",
        "precautions": [
            "Identify and minimize exposure to known allergens.",
            "Use over-the-counter antihistamines or nasal corticosteroid sprays.",
            "Keep indoor air clean by using HEPA filters and closing windows.",
            "Wash clothes and shower after spending time outdoors to remove pollen.",
            "Dust and vacuum your living spaces regularly."
        ]
    },
    "Migraine": {
        "description": "A neurological condition characterized by intense, throbbing headache pain, frequently accompanied by nausea and sensitivity to light and sound.",
        "severity": "Medium",
        "precautions": [
            "Rest in a quiet, darkened room at the onset of symptoms.",
            "Apply a cold compress to your forehead or the base of your skull.",
            "Avoid sensory triggers such as bright screens, loud noises, and strong smells.",
            "Stay hydrated, but avoid caffeine, chocolate, and alcohol.",
            "Take prescribed abortive medications or OTC pain relievers early in the attack."
        ]
    },
    "Pneumonia": {
        "description": "An infection that inflames the air sacs in one or both lungs, which may fill with fluid or pus, causing severe cough, fever, and difficulty breathing.",
        "severity": "High",
        "precautions": [
            "Consult a medical professional immediately; prescription medication is often required.",
            "Ensure absolute rest and avoid any physical exertion.",
            "Inhale steam or use a humidifier to help loosen thick chest mucus.",
            "Avoid smoking and exposure to secondary smoke.",
            "Monitor symptoms closely and seek emergency help if breathing becomes shallow or rapid."
        ]
    },
    "Dengue": {
        "description": "A mosquito-borne viral disease causing severe flu-like illness. It is characterized by high fever, intense joint/bone pain ('breakbone fever'), and skin rashes.",
        "severity": "High",
        "precautions": [
            "Prevent dehydration by drinking plenty of fluids (oral rehydration salts, water, juices).",
            "Take Acetaminophen for pain/fever relief; AVOID Ibuprofen, Aspirin, and Naproxen as they increase bleeding risk.",
            "Ensure complete bed rest in a screened room or under a mosquito net.",
            "Use mosquito repellents and wear long-sleeved clothing.",
            "Immediately visit a hospital if you develop warning signs like persistent vomiting or bleeding."
        ]
    },
    "Healthy": {
        "description": "Your symptom profile matches a healthy individual. No major disease signatures were detected by the machine learning model.",
        "severity": "Low",
        "precautions": [
            "Maintain a balanced diet rich in whole foods, vegetables, and proteins.",
            "Engage in at least 150 minutes of moderate physical activity weekly.",
            "Prioritize 7-9 hours of consistent sleep nightly.",
            "Wash hands regularly with soap and water to prevent pathogens.",
            "Schedule routine annual checkups with your primary care physician."
        ]
    }
}

DEFAULT_METADATA = {
    "description": "Symptom pattern matched to diagnostic profile.",
    "severity": "Medium",
    "precautions": ["Consult a doctor for advice."],
}
