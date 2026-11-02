# Naive Bayes spam filter
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

train = [
    ("Congratulations! You have won an iPhone, follow the link", "spam"),
    ("Urgent discount 90 percent only today buy now", "spam"),
    ("Your account is blocked, confirm your password via the link", "spam"),
    ("Get a loan without paperwork in 5 minutes", "spam"),
    ("A free prize is waiting, click here", "spam"),
    ("A reminder about the lab session on Thursday at 10:00", "mail"),
    ("Sending the lab report for review", "mail"),
    ("The department meeting is moved to Monday", "mail"),
    ("The consultation schedule before the exam is updated", "mail"),
    ("Please confirm your participation in the conference by Friday", "mail"),
]
texts, labels = zip(*train)
vec = CountVectorizer(lowercase=True)
# alpha=1: Laplace smoothing
nb = MultinomialNB(alpha=1.0).fit(vec.fit_transform(texts), labels)

for msg in ["You have won a free loan, click the link",
            "I will send the lab report by Thursday",
            "Confirm the password to your account today"]:
    p = nb.predict_proba(vec.transform([msg]))[0]
    print(f"P(spam) = {p[list(nb.classes_).index('spam')]:.3f}  <- {msg}")
