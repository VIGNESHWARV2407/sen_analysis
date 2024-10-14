const form = document.getElementById('sentiment-form');
const inputText = document.getElementById('input-text');
const checkBtn = document.getElementById('check-btn');
const resultDiv = document.getElementById('result');

checkBtn.addEventListener('click', async (e) => {
    e.preventDefault(); // prevent default form submission behavior
    const inputData = { inputs: inputText.value };
    try {
        const response = await query(inputData);
        const sentimentResult = response[0]; // assuming the response is an array with one object
        const maxScore = Math.max(...sentimentResult.map((item) => item.score));
        const maxLabel = sentimentResult.find((item) => item.score === maxScore).label;
        resultDiv.innerText = `Sentiment Analysis Result: ${maxLabel} (${maxScore.toFixed(2)})`;
    } catch (error) {
        console.error(error);
        resultDiv.innerText = 'Error occurred while processing the request.';
    }
});

async function query(data) {
    const response = await fetch(
        "https://api-inference.huggingface.co/models/cardiffnlp/twitter-roberta-base-sentiment-latest",
        {
            headers: {
                Authorization: "Bearer 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",
                "Content-Type": "application/json",
            },
            method: "POST",
            body: JSON.stringify(data),
        }
    );
    const result = await response.json();
    return result;
}