import pandas as pd
import re
import joblib
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import FunctionTransformer
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, classification_report
from sklearn.model_selection import train_test_split

# مرحلة التنضيف اللي الداتا هتعدي عليها قبل ما تتحلل
def clean_text_logic(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    # استبدال الرموز بمسافات
    text = re.sub(r'[-_/]', ' ', text)
    # إزالة علامات التقيم مع الحفاظ على الكلمات
    text = re.sub(r'[^\w\s]', '', text)
    # تنظيف المسافات الزائدة
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def text_cleaner_transformer(text_list):
    # الدالة دي اللي الـ Pipeline بيستخدمها لتحويل القائمة كاملة
    return pd.Series(text_list).apply(clean_text_logic)

# مرحلة تحميل الداتا عشان التدريب
try:
    data = pd.read_excel("final_cleaned_data.xlsx")
    x = data['clean_text']
    y = data['Category']
except FileNotFoundError:
    print("File final_cleaned_data.xlsx not found!")
    exit()

# مرحلة تقسيم الداتا للتدريب و الاختبار
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)


integrated_pipeline = Pipeline([
    ('cleaner', FunctionTransformer(text_cleaner_transformer)), # هنا خطوة التنضيف عشان الداتا تعدي عليها قبل التدريب او الاختبار
    ('tfidf', TfidfVectorizer(ngram_range=(1, 3), stop_words=None)),
    ('classifier', LinearSVC(C=1.0, random_state=42, max_iter=2000))
])
# تدريب الموديل
print('Training the LinearSVC model... please wait')
integrated_pipeline.fit(x_train, y_train)
y_pred = integrated_pipeline.predict(x_test)

# مرحلة تقييم الموديل عشان نعرف جودتة و طلعت 98
print('Model Evaluation')
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
print(f"Confusion: {confusion_matrix(y_test, y_pred)}")
print(f"Precision: {precision_score(y_test, y_pred, average='macro')}")
print(f"Recall: {recall_score(y_test, y_pred, average='macro')}")
print(f"Report: {classification_report(y_test, y_pred)}")

# حفظ الموديل النهائي للأبلكيشن في فايل جاهز بدل ما نكتب الكود كل مره ده فايل الموديل متدرب جاهز
joblib.dump(integrated_pipeline, 'incidentIQ_SVC.pkl')
print("Model Saved Successfully as 'incidentIQ_SVC.pkl'")
