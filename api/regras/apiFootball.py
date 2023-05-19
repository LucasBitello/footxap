import http
import datetime
from json import loads, dumps
from urllib.parse import quote

class RegraAPIFootBall:
    def conecarAPIFootball(self, params: str, isHOuveErroCOnexao: bool = False) -> list:
        url = "v3.football.api-sports.io"
        conexao = http.client.HTTPSConnection(url)
        headers = {
            'x-rapidapi-host': url,
            'x-rapidapi-key': "01d14b089686c4521d534e66b70e1a40"
        }
        urlParams = "/%s" % (params)
        newURLParams = quote(urlParams, safe=':/?&=')

        if newURLParams != urlParams:
            print("url normalizada by: " + "https://" + url + urlParams)
            urlParams = newURLParams
            print("url normalizada to: " + "https://" + url + urlParams)

        isDeuCertoRequest = False
        nroMaxTentativas = 5
        nroTentativas = 0
        print("\n ParametrosUrl: \n %s \n URL: \n %s" % (urlParams, ("https://" + url + urlParams)))

        while not isDeuCertoRequest and nroTentativas < nroMaxTentativas:
            nroTentativas += 1
            print("fez ", nroTentativas, " tentativa de conexao")

            conexao.request("GET", urlParams, headers=headers)

            resposta = conexao.getresponse()
            data = resposta.read()
            if resposta.status == 200:
                if data:
                    dataNormalizada = loads(data.decode("utf-8"))
                    isDeuCertoRequest = True
                else:
                    print("print erro na requisição porém status 200", data)
            else:
                print("print erro na requisição status: ", resposta.status, ", dados: ", data)


        return dataNormalizada