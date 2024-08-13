import os
from benfords import plot, benfords, test
import pandas as pd
import streamlit as st
from streamlit_navigation_bar import st_navbar
import pages as pg
import PyPDF2, re
import pycaret.anomaly as pa
import pycaret.clustering as pc
from io import BytesIO
from PIL import Image

st.set_page_config(
    page_title="Streamlit Navigation Bar Example 3",
    initial_sidebar_state="collapsed",
)


def get_unique_words_from_pdf(uploaded_file):
    pdf_reader = PyPDF2.PdfReader(uploaded_file)
    unique_words = set()

    for page_num in range(len(pdf_reader.pages)):
        page = pdf_reader.pages[page_num]
        text = page.extract_text()
        words = re.findall(r'\b\w+\b', text.lower())
        unique_words.update(words)

    return unique_words

def remove_escape_characters(string):
    pattern = r'\\[^\w\s]'
    return re.sub(pattern, '', string)

def search_word_pattern_in_pdf(file, pattern):
	pdf = PyPDF2.PdfReader(file)
	num_pages = len(pdf.pages)
	keyword_matches = []
	for page_number in range(num_pages):
		page = pdf.pages[page_number]
		text = page.extract_text()
		text = remove_escape_characters(text)
		matches = re.findall(pattern, text)
		if matches:
			for match in matches:
				keyword_matches.append(f"Pattern {pattern} found on page {page_number + 1}: {match}")

	return keyword_matches

def search_pdf_for_keywords(pdf_file, keyword):
	reader = PyPDF2.PdfReader(file)
	num_pages = len(reader.pages)
	pattern = r'\b{}\b'.format(re.escape(keyword))
	results = []

	for page_num in range(num_pages):
		page = reader.pages[page_num]
		text = page.extract_text()
		matches = re.findall(pattern, text, re.IGNORECASE)
		if matches:
			results.append(f"Page {page_num + 1}: {', '.join(matches)}")

	return results


def count_words_per_page(pdf_file):
    pdf = PyPDF2.PdfReader(pdf_file)
    num_pages = len(pdf.pages)
    word_counts = []

    for page_num in range(num_pages):
        page = pdf.pages[page_num]
        text = page.extract_text()
        words = text.split()
        unique_words = set(words)
        word_count = len(words)
        unique_word_count = len(unique_words)
        page_info = {
            "File Name": pdf_file.name,
            "Page Number": page_num + 1,
            "Number of words on the page": word_count,
            "Number of unique words on page": unique_word_count
        }
        word_counts.append(page_info)

    return word_counts

def count_keywords_in_pdf(pdf_file, keywords):
    pdf = PyPDF2.PdfReader(pdf_file)
    num_pages = len(pdf.pages)
    keyword_counts = []

    for page_num in range(num_pages):
        page = pdf.pages[page_num]
        text = page.extract_text()
        page_keywords = {}

        for keyword in keywords:
            count = text.lower().count(keyword.lower())
            if count > 0:
                page_keywords[keyword] = count

        if page_keywords:
            page_info = {
                "File Name": pdf_file.name,
                "Page Number": page_num + 1,
                "Keyword Counts": page_keywords
            }
            keyword_counts.append(page_info)

    return keyword_counts

def passed_failed_ocr_pages(pdf_file):
	pdf = PyPDF2.PdfReader(pdf_file)
	num_pages = len(pdf.pages)
	pages_without_text = {}
	pages_with_text = {}

	for page_num in range(num_pages):
		page = pdf.pages[page_num]
		text = page.extract_text()

		if not text.strip():
			page_info = {
				"File Name": pdf_file.name,
				"Page Number": page_num + 1
			}
			pages_without_text[page_num + 1] = page_info
		else:
			page_info = {
				"File Name": pdf_file.name,
				"Page Number": page_num + 1
			}
			pages_with_text[page_num + 1] = page_info

	return pages_without_text, pages_with_text


def extract_paragraphs_with_keywords(pdf_file, keywords):
    pdf = PyPDF2.PdfReader(pdf_file)
    num_pages = len(pdf.pages)
    extracted_paragraphs = []

    for page_num in range(num_pages):
        page = pdf.pages[page_num]
        text = page.extract_text()
        paragraphs = text.split("\n\n")

        for paragraph in paragraphs:
            for keyword in keywords:
                if keyword.lower() in paragraph.lower():
                    extracted_paragraphs.append({
                        "File Name": pdf_file.name,
                        "Page Number": page_num + 1,
                        "Keyword": keyword,
                        "Paragraph": paragraph.strip()
                    })

    return extracted_paragraphs

def display_pages_failed(pdf_file, pages):
	output_folder = rf'./{pdf_file.name}/Failed'
	extract_success = {}
	extract_failed = {}

	pdf = PyPDF2.PdfReader(pdf_file)

	if not os.path.exists(output_folder):
		os.makedirs(output_folder)

	for page_info in pages:
		page_num = int(page_info["Page Number"])
		if page_num <= len(pdf.pages):
			page = pdf.pages[page_num - 1]
			output_file = os.path.join(output_folder, f"{page_info['File Name']}_Page{page_num}.pdf")
			try:
				with open(output_file, "wb") as output:
					writer = PyPDF2.PdfFileWriter()
					writer.addPage(page)
					writer.write(output)
					extract_success[f"{page_info['File Name']}_Page{page_num}.pdf"] = "Success"
			except Exception as e:
				extract_failed[f"{page_info['File Name']}_Page{page_num}.pdf"] = "Fail"

	return extract_success, extract_failed


def read_pdf(file):
    pdf_reader = PyPDF2.PdfReader(file)
    num_pages = len(pdf_reader.pages)
    text = ""
    for page_num in range(num_pages):
        page = pdf_reader.pages[page_num]
        text += page.extract_text()
    return text

pages = ["Home", "Benfords Law", "Para-Thor", "Anomaly Detection", "Clustering", "About"]
parent_dir = os.path.dirname(os.path.abspath(__file__))
#logo_path = os.path.join(parent_dir, "PricewaterhouseCoopers_Logo.svg")

styles = {
	"nav": {
		"background-color": "#FFA500",
		"justify-content": "left",
		"height": "100px"
	},
	"span": {
		"padding": "75px",
	},
	"selected": {
		"background-color": "yellow",
		"color": "var(--text-color)",
		"font-weight": "normal",
		"padding": "14px",
	},
}

page = st_navbar(
    pages,
    selected="Home",
    #logo_path="",
    #urls=urls,
    styles=styles,
)

if page == "Home":
	st.write("\n")
	st.subheader("Home")



elif page == "Benfords Law":
	st.write("\n")
	style = "<style>.row-widget.stButton {text-align: center;}</style>"
	st.markdown(style, unsafe_allow_html=True)
	st.subheader("Benfords Law")
	st.write("\n")
	file = st.file_uploader("Upload your files here", key='benford_upload')
	if file is not None:
		if file.name.endswith('xlsx') or file.name.endswith("xls"):
			df = pd.read_excel(file)
			xls = pd.ExcelFile(file)
			sheet_names = xls.sheet_names
			sheet_audit = st.selectbox("Select a sheet to audit", options=sheet_names)
			if sheet_audit is not None:
				columns = df.columns
				column_to_audit = st.selectbox("Select a column to audit", options=columns)
				if column_to_audit is not None:
					audit_data = df[column_to_audit]
					audit_data = audit_data.astype(float).abs()
					start_number = st.number_input("Choose a digit to analyze", min_value=1, max_value=3, value=1)
					if st.button("Process"):
						results = benfords(
							data=audit_data,
							start_position=int(start_number),
							output_plot=True
						)
						st.write(results)
						if os.path.exists(r"C:\Users\ssolwa001\PycharmProjects\Audit_Keyword_Extractor\Frontend\benfords.jpg"):
							img = Image.open(r"C:\Users\ssolwa001\PycharmProjects\Audit_Keyword_Extractor\Frontend\benfords.jpg")
							st.image(img, caption="Benfords Law Analysis")
						else:
							st.warning("No Benfords Graph Generated!")

		elif file.name.endswith("csv"):
			df = pd.read_csv(file)
			columns = df.columns
			column_to_audit = st.selectbox("Select a column to audit", options=columns)
			if column_to_audit is not None:
				audit_data = df[column_to_audit]
				audit_data = audit_data.astype(float).abs()
				start_number = st.number_input("Choose a digit to analyze", min_value=1, max_value=3, value=1)
				if st.button("Process"):
					results = benfords(
						data=audit_data,
						start_position=int(start_number),
						output_plot=True
					)
					st.write(results)
					if os.path.exists(
							r"C:\Users\ssolwa001\PycharmProjects\Audit_Keyword_Extractor\Frontend\benfords.jpg"):
						img = Image.open(
							r"C:\Users\ssolwa001\PycharmProjects\Audit_Keyword_Extractor\Frontend\benfords.jpg")
						st.image(img, caption="Benfords Law Analysis")
					else:
						st.warning("No Benfords Graph Generated!")



elif page == "Para-Thor":
	st.write("\n")
	style = "<style>.row-widget.stButton {text-align: center;}</style>"
	st.markdown(style, unsafe_allow_html=True)
	st.subheader("Keyword Extractor")
	st.write("\n")
	file = st.file_uploader("Upload your files here", key='ParaThor_upload')
	grouped_words = []
	if file is not None:
		unique_words = get_unique_words_from_pdf(file)
		word_groups = st.text_input("Add groups of words here", placeholder="Separate each word by a comma. e.g. Word, Word2, Word3,...")
		word_groups = word_groups.split(",")
		for word in word_groups:
			grouped_words.append(word.lstrip().rstrip())
		keyword_select = st.multiselect("Select your keywords here", list(unique_words))
		if keyword_select is not None or keyword_select != "":
			for word in keyword_select:
				grouped_words.append(word)
		st.write(grouped_words)
		if grouped_words is not None or len(grouped_words) == 0:
			if st.button("Process"):
				st.subheader("Keyword Matches:\n")
				for keyword in grouped_words:
					st.write(keyword)
					pattern = rf'{keyword}'
					matches = search_word_pattern_in_pdf(file, pattern)
					#matches = search_pdf_for_keywords(file, pattern)
					st.write(matches)

				st.subheader("Word Counts per Page:\n")
				words_dict = count_words_per_page(file)
				st.write(words_dict)

				st.subheader("Keywords found in PDF:\n")
				keywords_in_pdf = count_keywords_in_pdf(file, keyword_select)
				st.write(keywords_in_pdf)

				st.subheader("Passed OCR pages:\n")
				failed, passed = passed_failed_ocr_pages(file)
				st.write(passed)

				st.subheader("Failed OCR pages:\n")
				st.write(failed)

				#success, fail = display_pages_failed(file, failed)

				st.subheader("Extracted Paragraphs per Keyword:\n")
				extracted_paragraphs = extract_paragraphs_with_keywords(file, keyword_select)
				st.write(extracted_paragraphs)


elif page == "Anomaly Detection":
	st.write("\n")
	style = "<style>.row-widget.stButton {text-align: center;}</style>"
	st.markdown(style, unsafe_allow_html=True)

	st.subheader("Anomaly Detection")
	st.write("\n")
	file = st.file_uploader("Upload your files here", key='Anomaly_Upload')
	if file is not None:
		if file.name.endswith('xlsx') or file.name.endswith("xls"):
			df = pd.read_excel(file)
			xls = pd.ExcelFile(file)
			sheet_names = xls.sheet_names
			sheet_audit = st.selectbox("Select a sheet to audit", options=sheet_names)
			column_drops = st.multiselect("Select columns that you do not want to include in the analysis",
										  options=df.columns)
			if column_drops is not None:
				df = df.drop(column_drops, axis=1)
			s = pa.setup(df, session_id=1234)
			models = ['abod', 'cluster', 'cof', 'iforest', 'histogram', 'knn', 'lof',
					  'svm', 'pca', 'mcd', 'sod', 'sos']
			select_model = st.selectbox("Choose a model", options=models)
			if select_model is not None:
				if st.button("Process"):
					model = pa.create_model(str(select_model))
					model_anomalies = pa.assign_model(model)
					pa.evaluate_model(model)
					predictions = pa.predict_model(model, data=df)
					st.dataframe(predictions)
					pa.plot_model(model, save=True)

		elif file.name.endswith('csv'):
			df = pd.read_csv(file)
			column_drops = st.multiselect("Select columns that you do not want to include in the analysis", options=df.columns)
			if column_drops is not None:
				df = df.drop(column_drops, axis=1)
			s = pa.setup(df, session_id=1234)
			models = ['abod', 'cluster', 'cof', 'iforest', 'histogram', 'knn', 'lof',
					  'svm', 'pca', 'mcd', 'sod', 'sos']
			select_model = st.selectbox("Choose a model", options=models)
			if select_model is not None:
				if st.button("Process"):
					model = pa.create_model(str(select_model))
					model_anomalies = pa.assign_model(model)
					pa.evaluate_model(model)
					predictions = pa.predict_model(model, data=df)
					st.dataframe(predictions)
					pa.plot_model(model, save=True)


elif page == "Clustering":
	st.write("\n")
	style = "<style>.row-widget.stButton {text-align: center;}</style>"
	st.markdown(style, unsafe_allow_html=True)

	st.subheader("Clustering")
	st.write("\n")
	file = st.file_uploader("Upload your files here", key="clustering")
	if file is not None:
		if file.name.endswith('xlsx') or file.name.endswith("xls"):
			df = pd.read_excel(file)
			xls = pd.ExcelFile(file)
			sheet_names = xls.sheet_names
			sheet_audit = st.selectbox("Select a sheet to audit", options=sheet_names)
			column_drops = st.multiselect("Select columns that you do not want to include in the analysis",
										  options=df.columns)
			if column_drops is not None:
				df = df.drop(column_drops, axis=1)

			models = ["kmeans", "ap", "meanshift", "sc", "hclust", "dbscan", "birch"]
			select_model = st.selectbox("Choose a model", options=models)
			if select_model is not None:
				if st.button("Process"):
					s = pc.setup(df, session_id=12345)
					model = pc.create_model(str(select_model))
					model_cluster = pc.assign_model(model)
					pc.evaluate_model(model)
					predictions = pc.predict_model(model, data=df)
					st.dataframe(predictions)
					pc.plot_model(model)


		elif file.name.endswith("csv"):
			df = pd.read_csv(file)
			column_drops = st.multiselect("Select columns that you do not want to include in the analysis",
										  options=df.columns)
			if column_drops is not None:
				df = df.drop(column_drops, axis=1)
			models = ["kmeans", "ap", "meanshift", "sc", "hclust", "dbscan",
					  "birch"]
			select_model = st.selectbox("Choose a model", options=models)
			if select_model is not None:
				if st.button("Process"):
					s = pc.setup(df, session_id=12345)
					model = pc.create_model(str(select_model))
					model_cluster = pc.assign_model(model)
					pc.evaluate_model(model)
					predictions = pc.predict_model(model, data=df)
					st.dataframe(predictions)
					pc.plot_model(model)


elif page == "About":
	st.subheader("About")

html = {
    "hide_sidebar_button": """
        <style>
            div[data-testid="collapsedControl"] {
                visibility: hidden;
            }
        </style>
    """,
}

st.markdown(html["hide_sidebar_button"], unsafe_allow_html=True)