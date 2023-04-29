async function callPOSTAPI(dadosPost, params) {
    let csrfmiddlewaretoken = getCookiecsrftoken()
    let url = URL_BASE_API + params
    console.log("fetching: " + url )

    let response = await fetch(url, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          "X-CSRFToken": csrfmiddlewaretoken
        },
        body: JSON.stringify(dadosPost),
    })

    return await response.json()
}

async function callGETAPI(params) {
    let loader = document.createElement('div');
    loader.classList.add('loader');
    document.body.appendChild(loader);

    let csrfmiddlewaretoken = getCookiecsrftoken()
    let url = URL_BASE_API + params
    console.log("Requisição para: " + url )

    let response = await fetch(url, {
        method: 'GET',
        headers: {
            'Content-Type': 'application/json',
            "X-CSRFToken": csrfmiddlewaretoken
        },
    })

    if (!response.ok){
        document.body.removeChild(loader);
    }

    let responseJson = await response.json()
    console.log("Retornou: \n")
    console.log(responseJson)
    document.body.removeChild(loader);
    return responseJson["response"]
}
