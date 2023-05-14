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

async function callGETAPI(params, isShowLoader = true) {
    let loader = document.createElement('div');

    if (isShowLoader){
        loader.classList.add('loader');
        document.body.appendChild(loader);
    }


    let csrfmiddlewaretoken = getCookiecsrftoken()
    let url = URL_BASE_API + params
    console.log("Requisição para: " + url )

    try{
        let response = await fetch(url, {
            method: 'GET',
            headers: {
                'Content-Type': 'application/json',
                "X-CSRFToken": csrfmiddlewaretoken
            },
            timeout: 900
        })

        let responseJson = await response.json()
        console.log("Retornou: \n")
        console.log(responseJson)
        if (isShowLoader){
            document.body.removeChild(loader);
        }
        return responseJson["response"]
    }catch (error){
        if (isShowLoader){
            document.body.removeChild(loader);
        }
    }



}
