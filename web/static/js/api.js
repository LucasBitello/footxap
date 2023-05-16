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

async function callGETAPI(params, isShowLoader = true, isMostrarMensagem = false) {
    let loader = document.createElement('div');

    if (isMostrarMensagem){
        loader.innerHTML = `
            <label>${gerarMensagemAleatoria()}</label>
        `
    }


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
            timeout: 900000
        })

        let responseJson = await response.json()
        console.log("Retornou: \n")
        console.log(responseJson)
        if (isShowLoader){
            document.body.removeChild(loader);
        }
        if (responseJson["erro"] === undefined){
            return responseJson["response"]
        }else {
            alert(responseJson["erro"])
        }

    }catch (error){
        if (isShowLoader){
            document.body.removeChild(loader);
        }
    }
}

function gerarMensagemAleatoria(){
    let arrMensagens = []
    arrMensagens.push(" quando voltar pelo menos lave a mão.")
    arrMensagens.push(" prepare um café.")
    arrMensagens.push(" reflita sobre sua vida.")
    arrMensagens.push(" tira uma soneca.")
    arrMensagens.push(" pense nisso: se dois humberto estiverem juntos, posso chama-los de doisberto.")
    arrMensagens.push(" pense nisso: o maior inimigo da hipotenusa é o carteto fantástico")
    arrMensagens.push(" pense nisso: se eu cobrir uma papelada eu posso chama-la de pavestida.")

    let indexMensagem = Math.floor(Math.random() * (arrMensagens.length));
    let mensagemLoader = "Treinamento da rede pode levar até 10 minutos então " + arrMensagens[indexMensagem]
    return mensagemLoader
}
