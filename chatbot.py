from openai import AzureOpenAI
from flask import Flask, request, jsonify

client = AzureOpenAI(
    azure_endpoint='https://azure00.openai.azure.com/',
    api_version='2023-07-01-preview',
    api_key='b0aa353afad74497a56a318512400391'
)

app = Flask(__name__)

predefined_questions = {
    "Как мога да променя паролата си?": "За да промените паролата си, влезте в профила си и отидете в секцията 'Настройки на профила'. Намерете опцията 'Промяна на парола', въведете текущата си парола, след това новата парола и потвърдете новата парола. Запазете промените, за да актуализирате паролата си.",
    "Как да променя потребителското си име?": "За да промените потребителското си име, влезте в профила си и отидете в секцията 'Настройки на профила'. Намерете опцията 'Промяна на потребителско име', въведете новото си потребителско име и запазете промените.",
    "Защо е важно редовно да променям паролата си?": "Редовното променяне на паролата помага за подобряване на сигурността на акаунта ви, като намалява риска от неоторизиран достъп.",
    "Има ли указания за създаване на силна парола?": "Да, силната парола трябва да съдържа комбинация от главни и малки букви, цифри и специални символи.",
    "Мога ли да видя текущото си потребителско име или парола в настройките на акаунта?": "От съображения за сигурност текущата ви парола няма да бъде показана. Можете да видите потребителското си име в настройките на акаунта.",
    "Колко често трябва да променям паролата си?": "Препоръчва се да променяте паролата си на всеки 3 до 6 месеца.",
    "Мога ли да използвам една и съща парола за различни акаунти?": "Не е препоръчително да използвате една и съща парола за множество акаунти.",
    "Как мога да създам превод между моите акаунти?": "За да създадете превод между вашите акаунти, влезте в профила си и отидете в секцията 'Преводи'.",
    "Как мога да направя превод в същата банка?": "За да направите превод в същата банка, отидете в секцията 'Преводи'. Въведете името на получателя и IBAN-а, сумата за превод, описание и причина за транзакцията.",
    "Как мога да направя превод към друга банка?": "За да направите превод към друга банка, въведете името на получателя, IBAN-а, сумата и BIC на банката получател.",
    "Каква информация трябва да предоставя за превод между моите сметки?": "За превод между вашите сметки трябва да предоставите изходна сметка, целева сметка, сума, описание и причина.",
    "Какви данни са необходими за превод в същата банка?": "За превод в същата банка трябва да предоставите името на получателя, IBAN-а и сумата за превод.",
    "Каква допълнителна информация е необходима за превод към друга банка?": "За превод към друга банка трябва да предоставите BIC на банката получател.",
    "Мога ли да добавя описание и причина за превода си?": "Да, можете да добавите описание и причина за превода си. Тези полета са по избор.",
    "Как мога да прегледам историята на транзакциите си?": "За да прегледате историята на транзакциите си, влезте в профила си и отидете в секцията 'Сметки'.",
    "Как да създам акаунт?": "За да създадете акаунт, кликнете върху бутона 'Заяви сметка' на началната страница.",
    "Какво мога да правя с акаунта си?": "С акаунта си можете да преглеждате и управлявате детайлите на акаунта си, както и да изтегляте историята на транзакциите си.",
    "Как да прегледам детайлите на акаунта си?": "Влезте в профила си и отидете в секцията 'Средства', за да прегледате детайлите на акаунта си.",
    "Мога ли да изтегля историята на транзакциите си?": "Да, можете да изтеглите историята на транзакциите си чрез секцията 'Средства'.",
    "Как да променя името на акаунта си?": "За да промените името на акаунта си, влезте в профила си и отидете в секцията 'Средства' или 'Профил'.",
    "Мога ли да изтрия акаунта си?": "Да, можете да изтриете акаунта си. Това ще доведе до загуба на всички ваши данни.",
    "Какви видове данни се криптират в приложението?": "Криптираме чувствителни данни като лични идентификационни номера и финансова информация, за да осигурим вашата сигурност.",
    "Как се криптират данните ми?": "Използваме Advanced Encryption Standard (AES) с 256-битов ключ за криптиране на вашите данни.",
    "Какво представляват автоматичните удръжки и как работят?": "Автоматичните удръжки са плащания, които автоматично се изтеглят от вашата сметка на редовна основа, без да е необходимо ръчно действие. Тези удръжки обикновено включват месечни такси за сметката и погасяване на заеми.",
    "Как ще бъда уведомен за автоматичните удръжки?": "Ще получавате известия за автоматичните удръжки, като известие за месечната такса по сметката и известие за успешно погасяване на заема.",
    "Мога ли да прегледам подробностите за автоматичните удръжки?": "Да, можете да прегледате подробностите за автоматичните удръжки чрез историята на транзакциите, където ще видите датата, сумата и описанието на всяка транзакция."
}


def check_predefined_question(prompt):
    return predefined_questions.get(prompt)