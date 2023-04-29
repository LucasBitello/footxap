import http
import datetime
from json import loads
from urllib.parse import quote

class RegraAPIFootBall:
    def conecarAPIFootball(self, params: str) -> list:
        url = "v3.football.api-sports.io"
        conexao = http.client.HTTPSConnection(url)
        headers = {
            'x-rapidapi-host': url,
            'x-rapidapi-key': "{sua key api-sports aki}"
        }
        urlParams = "/%s" % (params)
        newURLParams = quote(urlParams, safe=':/?&=')

        if newURLParams != urlParams:
            print("url normalizada by: " + "https://" + url + urlParams)
            urlParams = newURLParams
            print("url normalizada to: " + "https://" + url + urlParams)

        conexao.request("GET", urlParams, headers=headers)
        print("\n ParametrosUrl: \n %s \n URL: \n %s" % (urlParams, ("https://" + url + urlParams)))

        resposta = conexao.getresponse()
        data = resposta.read()
        return loads(data.decode("utf-8"))
