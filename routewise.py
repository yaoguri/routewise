import streamlit as st
import os
import base64
import pandas as pd
import folium
import datetime
import requests
import re
import sys
import subprocess
from geopy.distance import great_circle
from streamlit_folium import folium_static
from streamlit_folium import st_folium
from langdetect import detect
from transformers import MarianMTModel, MarianTokenizer, pipeline
from fpdf import FPDF
from functools import lru_cache
import io

#try:
    #import sacremoses
#except ImportError:
    #subprocess.check_call([sys.executable, "-m", "pip", "install", "sacremoses"])

st.set_page_config(page_title='Itinerary Planner in Mendez, Cavite', layout="wide")

@st.cache_data
def load_csv(file_path):
    return pd.read_csv(file_path)

# Helper function to encode image to base64
def image_to_base64(file_path):
    with open(file_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode()

# Paths to images (adjust this based on your actual images)
static_folder = "static"
image_files = [
    "hardin.jpg", "St. Augustine Parish.JPG", "olaes.jpg", "paradizoo.jpg","Kaias Kitchen.jpg",
    "Yokis Farm.PNG","Louis Farm.jpg","Enchantes Farm.PNG","Noon Cafe.PNG","Tres Tesoras Private Resort.PNG",
    "Tita Lolas House.PNG","Kalmado Campsite.PNG","Juanas Private Resort.PNG","Mr. Diego Greenland Resort.PNG",
    "test.jpg","test.jpg","bukid_pamana.JPG","Loft Mendez.PNG","The Tapsishop.PNG",
    "Ejay and Ashlie Bulalohan Sizzling Steakhouse.PNG","Juddies Bulalo at Inihaw Express.PNG","Buenaventuras Garden Cafe.PNG","The Fern Gardens.JPG","Sanctuario Nature Farms.PNG",
    "test.jpg","test.jpg","test.jpg","test.jpg","test.jpg",
]

# Encode images
base64_images = [image_to_base64(os.path.join(static_folder, img)) for img in image_files]

# Descriptions for images (manually editable)
descriptions = [
    "Hardin de Asis - Hardin de Asis is a picturesque garden venue nestled in the serene town of Mendez, Cavite. Surrounded by lush greenery, it is the perfect location for weddings, receptions, and other special occasions. With its well-maintained landscaping, tranquil ambiance, and ample space, Hardin de Asis provides a stunning backdrop for memorable events. Its peaceful environment makes it a preferred choice for those seeking a relaxing escape close to nature.",
    "St. Augustine Parish - The St. Augustine Parish in Mendez, Cavite, is a historic Catholic church dedicated to Saint Augustine. Known for its timeless architecture and serene ambiance, the church has been a spiritual center for the local community for generations. Its elegant interior, highlighted by intricate details, makes it a popular choice for weddings and other religious ceremonies. The church is not only a place of worship but also a significant cultural landmark in the area.",
    "Olaes Resort - Olaes Resort is a family-friendly destination in Mendez, Cavite, offering a relaxing escape with its inviting pools and cozy cottages. Ideal for family outings, reunions, and group gatherings, the resort features well-kept facilities and a peaceful atmosphere. Whether you want to enjoy a refreshing swim or simply unwind in nature, Olaes Resort provides the perfect setting for relaxation and leisure..",
    "Paradizoo - Paradizoo is a unique theme park and zoo located in Mendez, Cavite, blending agricultural experiences with outdoor fun. It features a variety of farm animals, vibrant flower gardens, vegetable plots, and educational attractions that promote eco-friendly living. Visitors can enjoy feeding animals, learning about organic farming, or simply exploring the scenic landscapes. Paradizoo is an ideal destination for families, school trips, and nature enthusiasts.",
    "Kaia's Kitchen - Kaia’s Kitchen is a charming restaurant in Mendez, Cavite, offering a menu of Filipino comfort food made with love and fresh ingredients. Known for its cozy atmosphere and home-cooked flavors, it’s a favorite among locals and visitors alike. The warm hospitality and delightful meals make Kaia’s Kitchen a must-visit spot for casual dining and family meals.",
    "Yoki's Farm - Yoki’s Farm is a popular agri-tourism destination in Mendez, Cavite, showcasing sustainable farming practices, exotic plants, and a private collection of rare antiques. Guests can tour the farm, interact with animals, and marvel at the unique artifacts on display. The farm promotes eco-conscious living while providing an enriching and memorable experience for visitors of all ages.",
    "Louis Farm - Louis Farm offers a rustic countryside experience in Mendez, Cavite, where visitors can enjoy the beauty of nature and farm life. This family-friendly destination features fresh produce, interactive farm activities, and a serene atmosphere. Perfect for day trips or educational tours, Louis Farm connects guests to the joys of sustainable farming and rural living.",
    "Enchante's Farm - Enchante’s Farm is a peaceful retreat in Mendez, Cavite, ideal for those seeking to reconnect with nature. The farm provides a relaxing environment surrounded by scenic landscapes and fresh air. Visitors can enjoy leisurely walks, farm-to-table meals, and other outdoor activities. It’s a perfect spot for unwinding and experiencing the charm of countryside living.",
    "Noon Cafe - Noon Cafe is a quaint and stylish coffee shop in Mendez, Cavite, known for its specialty coffee, light snacks, and laid-back atmosphere. With its Instagram-worthy decor and friendly vibe, it’s a popular hangout for both locals and travelers. Whether you’re looking for a quiet place to work or a cozy spot to catch up with friends, Noon Cafe delivers a delightful experience.",
    "Tres Tesoras Private Villa - Tres Tesoras Private Villa is a luxurious getaway in Mendez, Cavite, offering exclusive accommodations for families and groups. With premium amenities such as a private pool, spacious rooms, and modern facilities, the villa is perfect for staycations, celebrations, or simply unwinding in style. Its privacy and tranquility make it a top choice for those seeking a secluded retreat.",
    "Tita Lola’s House - Tita Lola’s House is a nostalgic dining spot in Mendez, Cavite, that brings the charm of traditional Filipino cuisine and home-cooked meals. Set in a cozy, vintage-inspired setting, it offers a warm and welcoming ambiance that feels like dining at a beloved relative’s home. The authentic dishes and homey atmosphere make it a standout culinary destination.",
    "Kalmado Campsite - Kalmado Campsite offers a peaceful camping experience in Mendez, Cavite, surrounded by lush greenery and fresh mountain air. Perfect for outdoor enthusiasts, the campsite provides opportunities for hiking, bonfires, and stargazing. With its serene environment, it’s an excellent choice for reconnecting with nature and escaping the hustle of city life.",
    "Juana’s Private Resort - Juana’s Private Resort is a modern vacation spot in Mendez, Cavite, featuring well-maintained pools and comfortable accommodations. Ideal for families and small groups, the resort provides privacy and convenience for relaxing getaways, celebrations, or casual gatherings.",
    "Mr. Diego GreenLand Resort - Mr. Diego GreenLand Resort is a spacious leisure destination in Mendez, Cavite, offering swimming pools, picnic areas, and event spaces. The resort is ideal for family outings, team-building activities, and community celebrations, thanks to its wide-open spaces and relaxing ambiance.",
    "Chicks Ni Soy - Chicks Ni Soy is a popular eatery in Mendez, Cavite, known for its flavorful chicken dishes and affordable menu. It’s a favorite spot for casual dining, serving crispy fried chicken and Filipino comfort food.",
    "Chickoks Mendez - Chickoks Mendez is a local dining spot specializing in fried chicken and other quick and tasty Filipino meals. Its laid-back vibe and satisfying dishes make it a go-to place for a casual meal.",
    "Bukid Pamana - Bukid Pamana is a rustic farm destination in Mendez, Cavite, offering a mix of cultural heritage and eco-tourism experiences. Visitors can explore the farm’s natural beauty, enjoy local delicacies, and learn about traditional farming practices.",
    "Loft Mendez - Loft Mendez is a chic cafe in Mendez, Cavite, known for its artisanal coffee, delectable desserts, and stylish interiors. The cafe is a cozy retreat for coffee lovers and those seeking a relaxing space to unwind or socialize.",
    "The Tapsi Shop - The Tapsi Shop is a budget-friendly eatery in Mendez, Cavite, offering Filipino comfort food, including its signature tapa. It’s a favorite among locals for quick and satisfying meals.",
    "Ejay & Ashlie Bulalohan Sizzling Steakhouse - Ejay & Ashlie Bulalohan Sizzling Steakhouse is a dining destination in Mendez, Cavite, specializing in bulalo, sizzling plates, and other hearty Filipino dishes. Known for its flavorful food and welcoming atmosphere, it’s a must-visit for meat lovers.",
    "Juddies Bulalo at Inihaw Express - Juddies Bulalo at Inihaw Express serves traditional bulalo and grilled Filipino specialties in a casual setting. Its affordable menu and flavorful dishes make it a popular choice for locals and visitors alike.",
    "Buenaventura’s Garden Cafe - Buenaventura’s Garden Cafe is a charming dining spot in Mendez, Cavite, offering freshly brewed coffee, hearty meals, and homemade desserts in a garden-inspired setting. The cafe’s tranquil ambiance and delicious offerings make it a delightful place to relax.",
    "The Fern Gardens - The Fern Gardens is a lush garden venue in Mendez, Cavite, perfect for intimate gatherings, events, or a peaceful escape. With its beautiful landscaping and serene atmosphere, it’s a great place to unwind or celebrate special moments.",
    "Sanctuaria Nature Farms - Sanctuaria Nature Farms is an eco-friendly farm in Mendez, Cavite, promoting sustainable agriculture and organic farming. Visitors can enjoy educational tours, farm activities, and the fresh produce grown onsite.",
    "Palocpoc Church - Palocpoc Church is a quaint and historic Catholic church in Mendez, Cavite, offering a peaceful place for prayer and reflection. Its simplicity and cultural significance make it a cherished landmark in the local community.",
    "Test 26 - Sample description for image 28.",
    "Test 27 - Sample description for image 29.",
    "Test 28 - Sample description for image 30."
]

# CSS Styling (keeping original styling)
st.markdown("""
    <style>
    .image-container {
        text-align: center;
        margin-bottom: 20px;
    }
    .hover-image {
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        border-radius: 10px;
        cursor: pointer;
        width: 100%;
    }
    .hover-image:hover {
        transform: scale(1.05);
        box-shadow: 0px 4px 20px rgba(0, 0, 0, 0.3);
    }
    .image-caption {
        font-size: 14px;
        color: #ffff;
        margin-top: 8px;
    }

    body {
        font-family: 'Maison Neue', sans-serif;
    }
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Maison Neue', sans-serif;
    }
    div.stTabs [data-baseweb="tab"] {
        font-family: 'Maison Neue', sans-serif;
        font-size: 16px; /* Adjust font size */
        color: #FAFAFA; /* Text color */
        background-color: #0E1117; /* Background color */
        border: 1px solid #ddd; /* Border around each tab */
        padding: 10px; /* Adjust padding */
        border-radius: 8px; /* Rounded corners */
        margin-right: 5px; /* Space between tabs */
        cursor: pointer; /* Add pointer cursor */
    }
    div.stTabs [data-baseweb="tab"]:hover {
        background-color: #e0e0e0; /* Hover background */
        color: #0056b3; /* Hover text color */
    }
    div.stTabs [data-baseweb="tab"][aria-selected="true"] {
        background-color: #FF4B4B; /* Active tab background */
        color: white; /* Active tab text color */
        border: 1px solid #FF4B4B; /* Active tab border */
    }
    div.stButton button {
        height: 35px; /* Set your desired button height */
        width: 100%;
        max-width: 400px; /* Limit maximum width */
        min-width: 150px; /* Set a minimum width */
        text-align: center;
        padding: 10px;
        margin-top: 27px; /* Adjust the number of pixels */
    }
    </style>
    """, unsafe_allow_html=True)

st.title("Itinerary Planner in Mendez, Cavite")
st.text("Plan your perfect day in Mendez!")

# Load the DataFrame
static_folder = "static"
df = load_csv(os.path.join(static_folder, 'LOCATIONS1.csv'))
df.columns = df.columns.str.strip

# Ensure the DataFrame has the necessary columns
required_columns = ['Latitude', 'Longitude', 'Category', 'time_estimate', 'Location']
if not all(col in df.columns for col in required_columns):
    st.error("The data file must contain the following columns: " + ", ".join(required_columns))
    st.stop()

start2 = (14.115098, 120.910245)
start1 = (14.154888, 120.902931)

df['Category'] = df['Category'].str.split('|')
df['Category'] = df['Category'].apply(lambda x: x if isinstance(x, list) else [])

# Tabs
homeTab, spotsTab, mapTab = st.tabs(["Generate Itinerary", "Tourist Destinations", "Map of Mendez"])

def generate_detailed_itinerary(route, start_time, end_time, df):
    itinerary = []
    total_duration = (end_time - start_time).total_seconds() / 60  # Total minutes available
    num_locations = len(route)

    # Guard against empty route
    time_per_location = total_duration / num_locations if num_locations > 0 else 0  

    current_time = start_time

    for index, (lat, lon) in enumerate(route):
        # Find the destination name and category from the DataFrame
        location_match = df[(df['Latitude'] == lat) & (df['Longitude'] == lon)]
        if not location_match.empty:
            destination_name = location_match['Location'].values[0]
            category = location_match['Category'].values[0]  # Get the category
        else:
            destination_name = "Unknown Location"  # Handle case where location is not found
            category = "Unknown Category"

        destination_info = {
            "Destination": destination_name,
            "Coordinates": (lat, lon),
            "Start Time": current_time.strftime("%H:%M"),
            "End Time": (current_time + datetime.timedelta(minutes=time_per_location)).strftime("%H:%M"),
            "Category": category  # Include category instead of preferences
        }
        itinerary.append(destination_info)

        # Update current time for the next location
        current_time += datetime.timedelta(minutes=time_per_location)

    return itinerary

with homeTab:
    with st.form(key="generator_form"):
        col1, col2, col3, col4, col5 = st.columns(5)

        with col1: 
            startPoint = st.selectbox("San ka galing ?", ["North (Indang -Mendez Rd)", "South (Tagaytay-Nasugbu Hwy)"])
            routeStart = start1 if startPoint == "North (Indang -Mendez Rd)" else start2

        with col2:
            start_time = datetime.time(7, 0)
            timeStart = st.time_input("Target Start Time?", value=start_time)

        with col3:
            end_time = datetime.time(19, 0)
            timeEnd = st.time_input("Target End Time", value=end_time)
            if timeStart < start_time or timeStart > end_time:
                st.warning(f"The destinations might not be open during these hours. Please select a time between {start_time} and {end_time}.")
                timeStart = start_time

        with col4:
            all_categories = [
                'Food & Dining',
                'Stay & Accommodation',
                'Cafe',
                'Religious Sites',
                'Adventure & Activities',
                'Photography',
                'Nature & Outdoor Activities'
            ]
            preferences = st.multiselect("Preference (Will be prioritized)", all_categories)
            user_preferences = {category: 1 if category in preferences else 0 for category in all_categories}

        with col5:
            st.write("")
            st.write("")
            generate = st.form_submit_button("Generate", use_container_width=True)

    if generate:
        start_datetime = datetime.datetime.combine(datetime.date.today(), timeStart)
        end_datetime = datetime.datetime.combine(datetime.date.today(), timeEnd)
        total_duration = end_datetime - start_datetime
        total_available_time = total_duration.total_seconds() / 60  # Convert to minutes

        @lru_cache(maxsize=None)
        def calculate_distance_memoized(lat1, lon1, lat2, lon2):
            return great_circle((lat1, lon1), (lat2, lon2)).miles

        def interleave_preferences(start, destinations, user_preferences, preferred_categories, total_available_time):
            route = []
            current_location = start
            total_time = 0

            preferred_destinations = destinations[
                destinations['Category'].apply(lambda x: isinstance(x, list) and any(cat in x for cat in preferred_categories))
            ]

            other_destinations = destinations[
                destinations['Category'].apply(lambda x: isinstance(x, list) and not any(cat in x for cat in preferred_categories))
            ]

            preferred_destinations = preferred_destinations.to_dict('records')
            other_destinations = other_destinations.to_dict('records')

            while total_time < total_available_time and (preferred_destinations or other_destinations):
                for dest_list in [preferred_destinations, other_destinations]:
                    if dest_list:
                        # Greedy choice: Find the nearest destination
                        next_dest = min(
                        dest_list,
                        key=lambda x: calculate_distance_memoized(current_location[0], current_location[1], x['Latitude'], x['Longitude'])
                        )
                        print(f"Greedy choice: Adding {next_dest['Location']}")  # Log the choice
                        estimated_time = next_dest['time_estimate']

                        if total_time + estimated_time <= total_available_time:
                            route.append((next_dest['Latitude'], next_dest['Longitude']))
                            total_time += estimated_time
                            current_location = (next_dest['Latitude'], next_dest['Longitude'])
                            dest_list.remove(next_dest)
                        else:
                            dest_list.remove(next_dest)  # Remove if it doesn't fit in time

            return route, total_time

        
        route, total_time = interleave_preferences(routeStart, df, user_preferences, preferences, total_available_time)

        route_map = folium.Map(location=routeStart, zoom_start=12)

        full_route = [routeStart] + route  # Add the start point to the beginning of the route

        polyline_color = 'blue' if startPoint == start1 else 'green'
        # Draw the route as a polyline including the start point
        folium.PolyLine(locations=full_route, color=polyline_color, weight=5, opacity=0.7).add_to(route_map)

        # Extract the start point coordinates
        start_lat, start_lon = routeStart  # Use routeStart as the start point

        # Find the name of the start location
        start_location_match = df[(df['Latitude'] == start_lat) & (df['Longitude'] == start_lon)]
        start_destination_name = start_location_match['Location'].values[0] if not start_location_match.empty else "Start Point"

        # Mark the start point with a different color (e.g., red)
        folium.Marker(
            location=routeStart,
            popup='Start Point',
            icon=folium.Icon(color='red')  # Start point marked in red
        ).add_to(route_map)

        for index, (lat, lon) in enumerate(route):
            location_match = df[(df['Latitude'] == lat) & (df['Longitude'] == lon)]
            destination_name = location_match['Location'].values[0] if not location_match.empty else "Unknown Location"
            
            # Add a marker for each destination in the route
            folium.Marker(
                location=[lat, lon],
                popup=f'{destination_name}',
                icon=folium.Icon(color=polyline_color)  # Use the same color as the polyline
            ).add_to(route_map)

        st.success("Route optimization complete!")

        # Create the itinerary starting from the start point
        itinerary = []
        current_time = start_datetime

        # Add the start point as the first entry in the itinerary
        itinerary.append({
            "Destination": "Start Point",
            "Coordinates": routeStart,
            "Start Time": current_time.strftime("%H:%M"),
            "End Time": current_time.strftime("%H:%M"),  # No time spent at start
            "Category": "Starting Point"  # Indicate the start point category
        })

        time_per_location = total_time / len(route) if route else 0
        for lat, lon in route:
            location_match = df[(df['Latitude'] == lat) & (df['Longitude'] == lon)]
            if not location_match.empty:
                destination_name = location_match['Location'].values[0]
                category = location_match['Category'].values[0]
            else:
                destination_name = "Unknown Location"
                category = "Unknown Category"

            itinerary.append({
                "Destination": destination_name,
                "Coordinates": (lat, lon),
                "Start Time": current_time.strftime("%H:%M"),
                "End Time": (current_time + datetime.timedelta(minutes=time_per_location)).strftime("%H:%M"),
                "Category": category
            })

            current_time += datetime.timedelta(minutes=time_per_location)

        # Display detailed itinerary
        st.subheader("Your Detailed Itinerary")
        for item in itinerary:
            with st.expander(f"**{item['Destination']}**"):
                mapCol, picCol = st.columns(2)
                destination_lat, destination_lon = item['Coordinates']
                destination_map = folium.Map(location=[destination_lat, destination_lon], zoom_start=15)

                # Add a marker for the destination
                folium.Marker(
                    location=[destination_lat, destination_lon],
                    popup=item['Destination'],
                    icon=folium.Icon(color='blue')
                ).add_to(destination_map)

                # Display the Folium map in the mapCol
                with mapCol:
                    folium_static(destination_map)

                # Get the picture filename from the DataFrame
                picture_filename = df.loc[(df['Latitude'] == destination_lat) & (df['Longitude'] == destination_lon), 'pics'].values
                
                # Check if picture_filename is not empty and is a string
                if picture_filename.size > 0 and isinstance(picture_filename[0], str):
                    picture_path = os.path.join(static_folder, picture_filename[0])  # Get the first matching picture
                    with picCol:
                        st.image(picture_path, caption=item['Destination'], use_container_width=True)
                else:
                    with picCol:
                        st.write("No image available for this destination.")
            #st.write(f"**{item['Destination']}**")  # Display the destination name
            st.write(f"Coordinates: {item['Coordinates']}")
            st.write(f"Start Time: {item['Start Time']}")
            st.write(f"End Time: {item['End Time']}")
            st.write(f"Category: {item['Category']}")  # Display the category
            st.write("---")
            
        folium_static(route_map)


with spotsTab:
    def load_reviews(file_path="static\REVIEWS.csv"):
        try:
            reviews_df = pd.read_csv(file_path)
            return reviews_df
        except FileNotFoundError:
            st.error("CSV file not found. Please ensure the file exists.")
            return pd.DataFrame(columns=["location", "user", "review"])
    
    def get_reviews_by_location(reviews_df, location):
        return reviews_df[reviews_df["location"].str.lower() == location.lower()]
    
    def display_reviews(location, reviews_df):
        reviews = get_reviews_by_location(reviews_df, location)
        if reviews.empty:
            st.write(f"No reviews found for {location}.")
        else:
            for _, row in reviews.iterrows():
                st.write(f"**{row['user']}**: {row['review']}")

    reviews_data = load_reviews()

    @st.cache_data
    def load_sentiment_words(pos_file="static/positive-words.txt", neg_file="static/negative-words.txt"):
        try:
            with open(pos_file, "r", encoding="utf-8") as f:
                positive_words = set(word.strip().lower() for word in f if not word.startswith(";"))
            with open(neg_file, "r", encoding="utf-8") as f:
                negative_words = set(word.strip().lower() for word in f if not word.startswith(";"))
            return positive_words, negative_words
        except FileNotFoundError as e:
            st.error(f"File not found: {e.filename}. Ensure the file is in the correct directory.")
            return set(), set()
        except OSError as e:
            st.error(f"Error opening file: {e}")
            return set(), set()

    @st.cache_data
    def load_translation_models():
        model_names = {
            "Tagalog": "Helsinki-NLP/opus-mt-en-tl",
            "Cebuano": "Helsinki-NLP/opus-mt-en-ceb",
            "Iloko": "Helsinki-NLP/opus-mt-en-ilo"

        }
        models = {}
        for lang, model_name in model_names.items():
            tokenizer = MarianTokenizer.from_pretrained(model_name)
            model = MarianMTModel.from_pretrained(model_name)
            models[lang] = (tokenizer, model)
        return models
    
    def load_translation_model():
        model_name = "Helsinki-NLP/opus-mt-tl-en"
        tokenizer = MarianTokenizer.from_pretrained(model_name)
        model = MarianMTModel.from_pretrained(model_name)
        return tokenizer, model
    
    def translate_text(text, tokenizer, model):
        batch = tokenizer.prepare_seq2seq_batch([text], return_tensors="pt")
        translated = model.generate(**batch)
        return tokenizer.decode(translated[0], skip_special_tokens=True)
    
    def translate_reviews(reviews, lang, translation_models):
        if len(reviews) == 1:  # Base case
            return [translate_text_to_language(reviews[0], lang, translation_models)]
        else:
            mid = len(reviews) // 2
            left = translate_reviews(reviews[:mid], lang, translation_models)  # Divide
            right = translate_reviews(reviews[mid:], lang, translation_models)  # Conquer
            return left + right

    def translate_text_to_language(text, lang, translation_models):
        if lang in translation_models:
            tokenizer, model = translation_models[lang]
            batch = tokenizer.prepare_seq2seq_batch([text], return_tensors="pt")
            translated = model.generate(**batch)
            return tokenizer.decode(translated[0], skip_special_tokens=True)
        return text    
    
    def analyze_sentiment(review, positive_words, negative_words):
        review_words = re.findall(r'\w+', review.lower())
        positive_count = sum(1 for word in review_words if word in positive_words)
        negative_count = sum(1 for word in review_words if word in negative_words)

        if positive_count > negative_count:
            return "Positive"
        elif negative_count > positive_count:
            return "Negative"
        else: 
            return "Neutral"

    def is_text_english(text):
        try:
            language = detect(text)
            return language == "en"
        except Exception:
            return False
        
    def display_reviews_with_translation(location, reviews_df, translation_models, positive_words, negative_words):
        reviews = get_reviews_by_location(reviews_df, location)
        if reviews.empty:
            st.write(f"No reviews found for {location}.")
        else:
            for _, row in reviews.iterrows():
                user, review = row["user"], row["review"]
                st.write(f"**{user}**: {review}")

                # Check if translation is needed
                if not is_text_english(review):
                    st.write("Non-English review detected.")
                    translated_review = translate_text_to_language(review, "English", translation_models)
                    st.write(f"Translated to English: _{translated_review}_")
                else:
                    translated_review = review

                # Sentiment analysis
                sentiment = analyze_sentiment(translated_review, positive_words, negative_words)
                st.write(f"**Sentiment**: {sentiment}")

                # Provide translation options
                lang_choice = st.selectbox(
                    "Translate to another language:",
                    ["None", "Tagalog", "Cebuano", "Iloko"],
                    key=f"{user}_{_}"
                )
                if lang_choice != "None":
                    translated_reviews = translate_reviews([review], lang_choice, translation_models)
                    st.write(f"Translated to {lang_choice}: _{translated_reviews[0]}_")


                st.write("---")

    # Load data and models
    positive_words, negative_words = load_sentiment_words()
    translation_models = load_translation_models()
    reviews_data = load_reviews()

    # Define a function to display the image details
    def display_image_details(image_index):
        st.markdown(f"""
            <div style="text-align: center;">
                <img src="data:image/jpeg;base64,{base64_images[image_index]}" style="width: 60%; border-radius: 10px;">
                <h3 style="color: white;">{descriptions[image_index]}</h3>
            </div>
        """, unsafe_allow_html=True)

    # Helper function to create a "Back" button
    def back_to_main_page():
        st.session_state["current_page"] = "main"

    # Initialize session state
    if "current_page" not in st.session_state:
        st.session_state["current_page"] = "main"

    if "selected_image" not in st.session_state:
        st.session_state["selected_image"] = None

    # Logic for displaying either the main page or image details
    if st.session_state["current_page"] == "main":
        # Manually define each button and image separately
        col1, col2, col3 = st.columns(3)

        with col1:

            st.markdown(f"""
                <div class="image-container">
                    <img src="data:image/jpeg;base64,{base64_images[0]}" class="hover-image">
                    <p class="image-caption">{descriptions[0]}</p>
                </div>
            """, unsafe_allow_html=True)


            if st.button("Hardin de Asis Reviews"):
                st.session_state["selected_image"] = 0
                st.session_state["current_page"] = "image_details"

        with col2:

            st.markdown(f"""
                <div class="image-container">
                    <img src="data:image/jpeg;base64,{base64_images[1]}" class="hover-image">
                    <p class="image-caption">{descriptions[1]}</p>
                </div>
            """, unsafe_allow_html=True)
            if st.button("St. Augustine Parish Reviews"):
                st.session_state["selected_image"] = 1
                st.session_state["current_page"] = "image_details"

        with col3:

            st.markdown(f"""
                <div class="image-container">
                    <img src="data:image/jpeg;base64,{base64_images[2]}" class="hover-image">
                    <p class="image-caption">{descriptions[2]}</p>
                </div>
            """, unsafe_allow_html=True)
            if st.button("Olaes Resort Reviews"):
                st.session_state["selected_image"] = 2
                st.session_state["current_page"] = "image_details"

        # You can continue this pattern for all other images and buttons:
        col1, col2, col3 = st.columns(3)

        with col1:

            st.markdown(f"""
                <div class="image-container">
                    <img src="data:image/jpeg;base64,{base64_images[3]}" class="hover-image">
                    <p class="image-caption">{descriptions[3]}</p>
                </div>
            """, unsafe_allow_html=True)
            if st.button("Paradizoo Reviews"):
                st.session_state["selected_image"] = 3
                st.session_state["current_page"] = "image_details"

        with col2:

            st.markdown(f"""
                <div class="image-container">
                    <img src="data:image/jpeg;base64,{base64_images[4]}" class="hover-image">
                    <p class="image-caption">{descriptions[4]}</p>
                </div>
            """, unsafe_allow_html=True)
            if st.button("Kaia's Kitchen Reviews"):
                st.session_state["selected_image"] = 4
                st.session_state["current_page"] = "image_details"

        with col3:

            st.markdown(f"""
                <div class="image-container">
                    <img src="data:image/jpeg;base64,{base64_images[5]}" class="hover-image">
                    <p class="image-caption">{descriptions[5]}</p>
                </div>
            """, unsafe_allow_html=True)
            if st.button("Yoki's Farm Reviews"):
                st.session_state["selected_image"] = 5
                st.session_state["current_page"] = "image_details"


        col1, col2, col3 = st.columns(3)

        with col1:

            st.markdown(f"""
                <div class="image-container">
                    <img src="data:image/jpeg;base64,{base64_images[6]}" class="hover-image">
                    <p class="image-caption">{descriptions[6]}</p>
                </div>
            """, unsafe_allow_html=True)
            if st.button("Louis Farm Reviews"):
                st.session_state["selected_image"] = 6
                st.session_state["current_page"] = "image_details"

        with col2:

            st.markdown(f"""
                <div class="image-container">
                    <img src="data:image/jpeg;base64,{base64_images[7]}" class="hover-image">
                    <p class="image-caption">{descriptions[7]}</p>
                </div>
            """, unsafe_allow_html=True)
            if st.button("Enchante's Farm Reviews"):
                st.session_state["selected_image"] = 7
                st.session_state["current_page"] = "image_details"

        with col3:

            st.markdown(f"""
                <div class="image-container">
                    <img src="data:image/jpeg;base64,{base64_images[8]}" class="hover-image">
                    <p class="image-caption">{descriptions[8]}</p>
                </div>
            """, unsafe_allow_html=True)
            if st.button("Noon Cafe Reviews"):
                st.session_state["selected_image"] = 8
                st.session_state["current_page"] = "image_details"

        col1, col2, col3 = st.columns(3)

        with col1:

            st.markdown(f"""
                <div class="image-container">
                    <img src="data:image/jpeg;base64,{base64_images[9]}" class="hover-image">
                    <p class="image-caption">{descriptions[9]}</p>
                </div>
            """, unsafe_allow_html=True)
            if st.button("Tres Tesoras Private Villa Reviews"):
                st.session_state["selected_image"] = 9
                st.session_state["current_page"] = "image_details"

        with col2:

            st.markdown(f"""
                <div class="image-container">
                    <img src="data:image/jpeg;base64,{base64_images[10]}" class="hover-image">
                    <p class="image-caption">{descriptions[10]}</p>
                </div>
            """, unsafe_allow_html=True)
            if st.button("Tita Lola's House Reviews"):
                st.session_state["selected_image"] = 10
                st.session_state["current_page"] = "image_details"

        with col3:

            st.markdown(f"""
                <div class="image-container">
                    <img src="data:image/jpeg;base64,{base64_images[11]}" class="hover-image">
                    <p class="image-caption">{descriptions[11]}</p>
                </div>
            """, unsafe_allow_html=True)
            if st.button("Kalmado Campsite Reviews"):
                st.session_state["selected_image"] = 11
                st.session_state["current_page"] = "image_details"

        col1, col2, col3 = st.columns(3)

        with col1:

            st.markdown(f"""
                <div class="image-container">
                    <img src="data:image/jpeg;base64,{base64_images[12]}" class="hover-image">
                    <p class="image-caption">{descriptions[12]}</p>
                </div>
            """, unsafe_allow_html=True)
            if st.button("Juana's Private Resort Reviews"):
                st.session_state["selected_image"] = 12
                st.session_state["current_page"] = "image_details"

        with col2:

            st.markdown(f"""
                <div class="image-container">
                    <img src="data:image/jpeg;base64,{base64_images[13]}" class="hover-image">
                    <p class="image-caption">{descriptions[13]}</p>
                </div>
            """, unsafe_allow_html=True)
            if st.button("Mr. Diego GreenLand Resort Reviews"):
                st.session_state["selected_image"] = 13
                st.session_state["current_page"] = "image_details"

        with col3:

            st.markdown(f"""
                <div class="image-container">
                    <img src="data:image/jpeg;base64,{base64_images[14]}" class="hover-image">
                    <p class="image-caption">{descriptions[14]}</p>
                </div>
            """, unsafe_allow_html=True)
            if st.button("Chicks Ni Soy Reviews"):
                st.session_state["selected_image"] = 14
                st.session_state["current_page"] = "image_details"

        col1, col2, col3 = st.columns(3)

        with col1:

            st.markdown(f"""
                <div class="image-container">
                    <img src="data:image/jpeg;base64,{base64_images[15]}" class="hover-image">
                    <p class="image-caption">{descriptions[15]}</p>
                </div>
            """, unsafe_allow_html=True)
            if st.button("Chikoks Mendez Reviews"):
                st.session_state["selected_image"] = 15
                st.session_state["current_page"] = "image_details"

        with col2:

            st.markdown(f"""
                <div class="image-container">
                    <img src="data:image/jpeg;base64,{base64_images[16]}" class="hover-image">
                    <p class="image-caption">{descriptions[16]}</p>
                </div>
            """, unsafe_allow_html=True)
            if st.button("Bukid Pamana Reviews"):
                st.session_state["selected_image"] = 16
                st.session_state["current_page"] = "image_details"

        with col3:

            st.markdown(f"""
                <div class="image-container">
                    <img src="data:image/jpeg;base64,{base64_images[17]}" class="hover-image">
                    <p class="image-caption">{descriptions[17]}</p>
                </div>
            """, unsafe_allow_html=True)
            if st.button("Loft Mendez Reviews"):
                st.session_state["selected_image"] = 17
                st.session_state["current_page"] = "image_details"

        col1, col2, col3 = st.columns(3)

        with col1:

            st.markdown(f"""
                <div class="image-container">
                    <img src="data:image/jpeg;base64,{base64_images[18]}" class="hover-image">
                    <p class="image-caption">{descriptions[18]}</p>
                </div>
            """, unsafe_allow_html=True)
            if st.button("The Tapsi Shop Reviews"):
                st.session_state["selected_image"] = 18
                st.session_state["current_page"] = "image_details"

        with col2:

            st.markdown(f"""
                <div class="image-container">
                    <img src="data:image/jpeg;base64,{base64_images[19]}" class="hover-image">
                    <p class="image-caption">{descriptions[19]}</p>
                </div>
            """, unsafe_allow_html=True)
            if st.button("Ejay & Ashlie Bulalohan Sizzling Steakhouse Reviews"):
                st.session_state["selected_image"] = 19
                st.session_state["current_page"] = "image_details"

        with col3:

            st.markdown(f"""
                <div class="image-container">
                    <img src="data:image/jpeg;base64,{base64_images[20]}" class="hover-image">
                    <p class="image-caption">{descriptions[20]}</p>
                </div>
            """, unsafe_allow_html=True)
            if st.button("Juddies Bulalo at Inihaw Express Reviews"):
                st.session_state["selected_image"] = 20
                st.session_state["current_page"] = "image_details"

        col1, col2, col3 = st.columns(3)

        with col1:

            st.markdown(f"""
                <div class="image-container">
                    <img src="data:image/jpeg;base64,{base64_images[21]}" class="hover-image">
                    <p class="image-caption">{descriptions[21]}</p>
                </div>
            """, unsafe_allow_html=True)
            if st.button("Buenaventura's Garden Cafe Reviews"):
                st.session_state["selected_image"] = 21
                st.session_state["current_page"] = "image_details"

        with col2:

            st.markdown(f"""
                <div class="image-container">
                    <img src="data:image/jpeg;base64,{base64_images[22]}" class="hover-image">
                    <p class="image-caption">{descriptions[22]}</p>
                </div>
            """, unsafe_allow_html=True)
            if st.button("The Fern Gardens Reviews"):
                st.session_state["selected_image"] = 22
                st.session_state["current_page"] = "image_details"

        with col3:

            st.markdown(f"""
                <div class="image-container">
                    <img src="data:image/jpeg;base64,{base64_images[23]}" class="hover-image">
                    <p class="image-caption">{descriptions[23]}</p>
                </div>
            """, unsafe_allow_html=True)
            if st.button("Sanctuario Nature Farms Reviews"):
                st.session_state["selected_image"] = 23
                st.session_state["current_page"] = "image_details"

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown(f"""
                <div class="image-container">
                    <img src="data:image/jpeg;base64,{base64_images[24]}" class="hover-image">
                    <p class="image-caption">{descriptions[24]}</p>
                </div>
            """, unsafe_allow_html=True)
            if st.button("Palocpoc Chruch Reviews"):
                st.session_state["selected_image"] = 24
                st.session_state["current_page"] = "image_details"
        
        with col2:
            st.write("")
        
        with col3:
            st.write("")

    elif st.session_state["current_page"] == "image_details":
        selected_image_index = st.session_state["selected_image"]

        if selected_image_index is not None:
            display_image_details(selected_image_index)
        else:
            st.error("No image selected!")

        with st.expander("Reviews"):
            if selected_image_index == 0:
                display_reviews_with_translation("Hardin de Asis", reviews_data, translation_models, positive_words, negative_words)
            elif selected_image_index == 1:
                display_reviews_with_translation("St. Augustine Parish", reviews_data, translation_models, positive_words, negative_words)
            elif selected_image_index == 2:
                display_reviews_with_translation("Olaes Resort", reviews_data, translation_models, positive_words, negative_words)
            elif selected_image_index == 3:
                display_reviews_with_translation("Paradizoo", reviews_data, translation_models, positive_words, negative_words)
            elif selected_image_index == 4:
                display_reviews_with_translation("Kaia's Kitchen", reviews_data, translation_models, positive_words, negative_words)
            elif selected_image_index == 5:
                display_reviews_with_translation("Yoki's Farm", reviews_data, translation_models, positive_words, negative_words)
            elif selected_image_index == 6:
                display_reviews_with_translation("Louis Farm", reviews_data, translation_models, positive_words, negative_words)
            elif selected_image_index == 7:
                display_reviews_with_translation("Enchante's Farm", reviews_data, translation_models, positive_words, negative_words)
            elif selected_image_index == 8:
                display_reviews_with_translation("Noon Cafe", reviews_data, translation_models, positive_words, negative_words)
            elif selected_image_index == 9:
                display_reviews_with_translation("Tres Tesoras Private Villa", reviews_data, translation_models, positive_words, negative_words)
            elif selected_image_index == 10:
                display_reviews_with_translation("Tita Lola's House", reviews_data, translation_models, positive_words, negative_words)
            elif selected_image_index == 11:
                display_reviews_with_translation("Kalmado Campsite", reviews_data, translation_models, positive_words, negative_words)
            elif selected_image_index == 12:
                display_reviews_with_translation("Juana's Private Resort", reviews_data, translation_models, positive_words, negative_words)
            elif selected_image_index == 13:
                display_reviews_with_translation("Mr. Diego Greenland Resort", reviews_data, translation_models, positive_words, negative_words)
            elif selected_image_index == 14:
                display_reviews_with_translation("Chicks Ni Soy", reviews_data, translation_models, positive_words, negative_words)
            elif selected_image_index == 15:
                display_reviews_with_translation("Chikoks Mendez", reviews_data, translation_models, positive_words, negative_words)
            elif selected_image_index == 16:
                display_reviews_with_translation("Bukid Pamana", reviews_data, translation_models, positive_words, negative_words)
            elif selected_image_index == 17:
                display_reviews_with_translation("Loft Mendez", reviews_data, translation_models, positive_words, negative_words)
            elif selected_image_index == 18:
                display_reviews_with_translation("The Tapsi Shop", reviews_data, translation_models, positive_words, negative_words)
            elif selected_image_index == 19:
                display_reviews_with_translation("Ejay & Ashlie Bulalohan Sizzling Steakhouse", reviews_data, translation_models, positive_words, negative_words)
            elif selected_image_index == 20:
                display_reviews_with_translation("Juddies Bulalo at Inihaw Express", reviews_data, translation_models, positive_words, negative_words)
            elif selected_image_index == 21:
                display_reviews_with_translation("Buenaventura's Garden Cafe", reviews_data, translation_models, positive_words, negative_words)
            elif selected_image_index == 22:
                display_reviews_with_translation("The Fern Gardens", reviews_data, translation_models, positive_words, negative_words)
            elif selected_image_index == 23:
                display_reviews_with_translation("Sanctuario Nature Farms", reviews_data, translation_models, positive_words, negative_words)
            elif selected_image_index == 24:
                display_reviews_with_translation("Palocpoc Church", reviews_data, translation_models, positive_words, negative_words)


        if st.button("Back"):
            back_to_main_page()

with mapTab:
    # Streamlit app title
    st.title("MENDEZ , CAVITE")

    # Create a Folium map
    m = folium.Map(location=[14.1306245, 120.894918], zoom_start=13)  # Centered on Mendez

    # Add a marker to the map
    folium.Marker(
    location=[14.1306245, 120.894918],
    popup="London",
    icon=folium.Icon(color="blue", icon="info-sign"),
    ).add_to(m)

    # Display the Folium map in Streamlit
    st_folium(m, width=700, height=500)


