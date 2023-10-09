document.addEventListener('DOMContentLoaded', async () => {
    ajustarLayout()
    document.getElementById("select-country").addEventListener("change", async () => {
        await ajustarSelectLeague("select-league")

        localStorage.setItem("id_country_selected", document.getElementById("select-country").value)
        document.getElementById("select-league").value = localStorage.getItem("id_league_selected")

        let newEvent = new Event("change")
        document.getElementById("select-league").dispatchEvent(newEvent)
    })

    document.getElementById("select-league").addEventListener("change", async () => {
        let id_league = document.getElementById("select-league").value

        if(!isNaN(parseInt(id_league))){
            await ajustarSelectSeason("select-season")
            localStorage.setItem("id_league_selected", id_league)
            document.getElementById("select-season").value = localStorage.getItem("id_season_selected")
            let newEvent = new Event("change")
            document.getElementById("select-season").dispatchEvent(newEvent)
        }

    })

    document.getElementById("select-season").addEventListener("change", async () => {
        let id_season = document.getElementById("select-season").value

        if(!isNaN(parseInt(id_season))){
            localStorage.setItem("id_season_selected", id_season)
            await ajustarGridTabelaOuJogos()
        }
    })

    document.getElementById("select-tabela-jogos").addEventListener("change", async () => {
        localStorage.setItem("id_tabela_jogos_selected", document.getElementById("select-tabela-jogos").value)
        await ajustarGridTabelaOuJogos()
    })

    await ajustarSelectCountry("select-country")
    await searchTeams(true)
    await searchTeams(false)

    document.getElementById("select-country").value = localStorage.getItem("id_country_selected")
    document.getElementById("select-tabela-jogos").value = localStorage.getItem("id_tabela_jogos_selected")
    let newEvent = new Event("change")
    document.getElementById("select-country").dispatchEvent(newEvent)
    document.getElementById("select-tabela-jogos").dispatchEvent(newEvent)
});

async function ajustarSelectCountry(id_html_select) {
    let arrCountries = await callGETAPI("/countries")
    let selectCountry = document.getElementById(id_html_select)

    selectCountry.innerHTML = `<option value="0" selected>Selecione um país...</option>`

    for (let country of arrCountries){
        selectCountry.innerHTML += `<option value="${country["id"]}">${country["name"]}</option>`
    }
}


async function ajustarSelectLeague(id_html_select) {
    let selectLeagues = document.getElementById(id_html_select)
    let idSelectCountry = document.getElementById("select-country").value

    if (!isVazia(idSelectCountry) && !isNaN(parseInt(idSelectCountry))){
        let arrLeagues = await callGETAPI("/leagues?id_country="+idSelectCountry)

        selectLeagues.innerHTML = `<option value="0" selected>Selecione uma liga...</option>`

        for (let league of arrLeagues){
            selectLeagues.innerHTML += `<option value="${league["id"]}" class="fa-solid fa-arrow-down">${league["name"]}</option>`
        }
    }
}

async function ajustarSelectSeason(id_html_select) {
    let selectSeason = document.getElementById(id_html_select)
    let idSelectLeague = document.getElementById("select-league").value

    if (!isVazia(idSelectLeague) && !isNaN(parseInt(idSelectLeague))){
        let arrSeasons = await callGETAPI("/seasons?id_league="+idSelectLeague)

        selectSeason.innerHTML = `<option value="0" selected>Selecione uma temporada...</option>`

        for (let season of arrSeasons){
            let isPossuiEstatisticas = season["has_statistics_fixtures"] === 1 ? '&#128405;' : '';
            selectSeason.innerHTML += `<option value="${season["id"]}">${season["year"]} ${isPossuiEstatisticas}</option>`
        }
    }
}

async function ajustarGridTabelaOuJogos(){
    let div_tabela_jogos = document.getElementById("div-tabela-jogos")
    //let loader = document.createElement('div')
    //loader.classList.add('loader');
    //div_tabela_jogos.appendChild(loader)
    //let elemento = document.querySelector(".menu-left")
    //document.querySelector(".loader").style.width = window.getComputedStyle(elemento).getPropertyValue("width")
    let id_season = document.getElementById("select-season").value

    let value_select_tabela_jogos = document.getElementById("select-tabela-jogos").value
    console.log(value_select_tabela_jogos)
    if (parseInt(value_select_tabela_jogos) === 1){
        await showTabela(id_season)
    }else if (parseInt(value_select_tabela_jogos) === 2)
        await showJogos(id_season)

    //div_tabela_jogos.removeChild(loader)
}

async function searchTeams(isHome){
    let name_diff_item = isHome ? "-home" : "-away"
    let nameIdElementDivSearch = "div-results-team"+name_diff_item
    let nameIdElementInputSearch = "input-search-team"+name_diff_item

    let divSearchTeam = document.getElementById(nameIdElementDivSearch)
    let iptSearchTeam = document.getElementById(nameIdElementInputSearch)
    divSearchTeam.style.width = iptSearchTeam.getBoundingClientRect().width + "px"


    iptSearchTeam.addEventListener("keyup", async () => {
        let sltSeason = document.getElementById("select-season")

        if(isNaN(parseInt(sltSeason.value))){
            return;
        }

        let strParams = `?name=${iptSearchTeam.value}&id_season=${sltSeason.value}`
        callGETAPI("/teams/search"+strParams, false).then((response) => {
            divSearchTeam.innerHTML = ""
            for(let team of response){
                divSearchTeam.innerHTML +=
                `<a href="#" class="outline-none a-team${name_diff_item}" data-id-team="${team["id"]}" 
                    data-logo-team="${team["logo"]}" data-name-team="${team["name"]}">
                    ${team["name"]} <img src="${team["logo"]}" alt="img_team">
                </a>`
            }

            let elementsResultsSearchTeam = document.querySelectorAll(".a-team"+name_diff_item)
            for (let element of elementsResultsSearchTeam){
                element.addEventListener("click", async () => {
                    let id_team = element.getAttribute("data-id-team")
                    let logo_team = element.getAttribute("data-logo-team")
                    let name_team = element.getAttribute("data-name-team")

                    iptSearchTeam.value = name_team
                    iptSearchTeam.setAttribute("data-id-team-selected", id_team)

                    setImgTeam(isHome, logo_team)

                    if(!isHome){
                        let id_team_home_selected = document.getElementById("input-search-team-home")
                            .getAttribute("data-id-team-selected")

                        if(!isVazia(id_team_home_selected)){
                            await fazerRequisicaoParaIA(sltSeason.value, id_team_home_selected, id_team)
                        }
                    }
                })
            }
        })
    })

    iptSearchTeam.addEventListener("click", () => {
        divSearchTeam.classList.remove("hidden")
        let newEvent = new Event("keyup")
        iptSearchTeam.dispatchEvent(newEvent)
    })

    iptSearchTeam.addEventListener("focusout", () => {
        setTimeout(() => {
            let idSelected = iptSearchTeam.getAttribute("data-id-team-selected")
            if(isVazia(idSelected)){
                iptSearchTeam.value = ""
            }
            divSearchTeam.classList.add("hidden")
        }, 250)
    })
}

async function showTabela(id_season){
    if (isNaN(parseInt(id_season))){
        return
    }
    document.getElementById("jogos-teams").classList.add("hidden")
    let tabelaTeam = document.getElementById("tabela-teams")
    tabelaTeam.classList.remove("hidden")
    let tabelaPontuacao = await  callGETAPI("/tabela?id_season="+id_season)

    tabelaTeam.innerHTML = `
        <thead>
            <tr class="tr-team-tabela text-align-center margin-vertical-5px">
                <th class="max-width-25 width-25 td-team">Nome</th>
                <th class="max-width-10 width-10 td-team" title="Pontos">Pt</th>
                <th class="max-width-10 width-10 td-team" title="Partidas jogadas">PJ</th>
                <th class="max-width-10 width-10 td-team" title="Saldo gols">SG</th>
                <th class="max-width-10 width-10 td-team" title="Gols marcados">GM</th>
                <th class="max-width-10 width-10 td-team" title="Gols sofridos">GS</th>
                <th class="max-width-25 width-25 td-team">Ultimos -></th>
            
            </tr>
        </thead><tbody id="tbody-tabela-teams" class=""></tbody>`

    let tbodyTabelaTeam = document.getElementById("tbody-tabela-teams")
    for (let team of tabelaPontuacao["arr_team_pontuacao"]){
        tbodyTabelaTeam.innerHTML += `
            <tr class="tr-team-tabela text-align-center paddaing-vertical-3px">
                <td class="max-width-25 width-25 td-team" title="${team["name_team"]}">${team["name_team"]}</td>
                <td class="max-width-10 width-10 td-team">${team["pontos"]}</td>
                <td class="max-width-10 width-10 td-team">${team["qtde_jogos"]}</td>
                <td class="max-width-10 width-10 td-team">${team["saldo_gols"]}</td>
                <td class="max-width-10 width-10 td-team">${team["qtde_gols_marcados"]}</td>
                <td class="max-width-25 width-10 td-team">${team["qtde_gols_sofridos"]}</td>
                <td class="max-width-25 width-25 td-team">
                    <div id="ultimos-team-${team["id_team"]}" class="display-flex-row-space-around"></div>
                </td>
            </tr>`

        let arr_ultimos_jogos = team["arr_resultados_ultimos_jogos"]
        let arr_last_jogos = arr_ultimos_jogos.slice(-team["qtde_resultados_ultimos_jogos"])
        let div_team_ultimos_jogos = document.getElementById(`ultimos-team-${team["id_team"]}`)

        for(let jogo of arr_last_jogos){
            if (jogo["is_winner"] === 1){
                div_team_ultimos_jogos.innerHTML +=
                    `<div><i class="fa-solid ${jogo["is_home"] === 1 ? "fa-house" : "fa-circle" } color-vitoria font-size-0-8-em"></i></div>`
            }else if (jogo["is_winner"] === 0){
                div_team_ultimos_jogos.innerHTML +=
                    `<div><i class="fa-solid ${jogo["is_home"] === 1 ? "fa-house" : "fa-circle" } color-derrota font-size-0-8-em"></i></div>`
            }else if (jogo["is_winner"] === null){
                div_team_ultimos_jogos.innerHTML +=
                    `<div><i class="fa-solid ${jogo["is_home"] === 1 ? "fa-house" : "fa-circle" } color-empate font-size-0-8-em"></i></div>`
            }

        }
    }
}

async function showJogos(id_season){
    if (isNaN(parseInt(id_season))){
        return
    }
    document.getElementById("tabela-teams").classList.add("hidden")
    let jogosTeam = document.getElementById("jogos-teams")
    jogosTeam.classList.remove("hidden")
    let tabelaJogos = await  callGETAPI("/jogos?id_season="+id_season)
    jogosTeam.innerHTML = ``

    for (let team of tabelaJogos["arr_next_jogos"]){
        if(isVazia(team["team_home"]) || isVazia(team["team_away"])){
            continue
        }

        jogosTeam.innerHTML += `
            <tr class="tr-team-tabela paddaing-vertical-5px">
                <td>
                    <a href="#" class="a-link-team-vs-team" 
                    data-id-team-home="${team["team_home"]["id_team"]}"
                    data-name-team-home="${team["team_home"]["info_team"]["name"]}"
                    data-logo-team-home="${team["team_home"]["info_team"]["logo"]}"
                    data-id-team-away="${team["team_away"]["id_team"]}"
                    data-name-team-away="${team["team_away"]["info_team"]["name"]}"
                    data-logo-team-away="${team["team_away"]["info_team"]["logo"]}"> 
                        ${team["team_home"]["name_team"]} <b>VS</b> ${team["team_away"]["name_team"]}
                    </a> 
                    <br> as: ${team["data_jogo"]}
                </td>
            </tr>`

        let elementsTeamVsTeam = document.querySelectorAll(".a-link-team-vs-team")
        for (let element of elementsTeamVsTeam){
            element.addEventListener("click", async () => {
                let id_season_selected = document.getElementById("select-season").value
                let iptSearchTeamHome = document.getElementById("input-search-team-home")
                let iptSearchTeamAway = document.getElementById("input-search-team-away")

                let id_team_home = element.getAttribute("data-id-team-home")
                let name_team_home = element.getAttribute("data-name-team-home")
                let logo_team_home = element.getAttribute("data-logo-team-home")

                let id_team_away = element.getAttribute("data-id-team-away")
                let name_team_away = element.getAttribute("data-name-team-away")
                let logo_team_away = element.getAttribute("data-logo-team-away")

                iptSearchTeamHome.value = name_team_home
                iptSearchTeamAway.value = name_team_away

                setImgTeam(true, logo_team_home)
                setImgTeam(false, logo_team_away)


                await fazerRequisicaoParaIA(id_season_selected, id_team_home, id_team_away)
            })
        }
    }
}

async function fazerRequisicaoParaIA(id_season, id_team_home, id_team_away){
    document.getElementById("div-estatisticas-team-home").innerHTML = ``
    document.getElementById("div-estatisticas-team-away").innerHTML = ``
    document.getElementById("div-previsao-partida-team-home").innerHTML = ``
    document.getElementById("div-previsao-partida-team-away").innerHTML = ``
    document.getElementById("div-previsao-team-home").innerHTML = ``;
    document.getElementById("div-previsao-team-away").innerHTML = ``;

    await fazerRequisicaoEstatisticas("div-estatisticas-team-home", id_season, id_team_home, true)
    await fazerRequisicaoEstatisticas("div-estatisticas-team-away", id_season, id_team_away, false)
    //await fazerRequisicaoParaIAPreverTime("div-previsao-team-home", id_season, id_team_home, true)
    //await fazerRequisicaoParaIAPreverTime("div-previsao-team-away", id_season, id_team_away, false)
    await fazerRequisicaoParaIAPreverPartida(id_season, id_team_home, id_team_away)
}

async function fazerRequisicaoParaIAPreverPartida(id_season, id_team_home, id_team_away){
    let div_estatisticas_team_home = document.getElementById("div-previsao-partida-team-home")
    let div_estatisticas_team_away = document.getElementById("div-previsao-partida-team-away")

    let params = "/previsao-partida?id_season="+id_season+"&id_team_home="+id_team_home+"&id_team_away="+id_team_away
    let probsIA = await callGETAPI(params, true, true)

    div_estatisticas_team_home.innerHTML = `
        <div class="div-info-resultados-ia">
            <label>Previsões geradas pela IA versão: ${probsIA["v_ia"]} </label><br>
            <label>Erro da rede ficou em: ${probsIA["erro"]} </label><br>
            <label>Foi usado os ultimos ${probsIA["qtde_dados"]} jogos desse time. </label><br>
            <label>Previsão pra o jogo do dia: ${probsIA["data_jogo_previsto"]}</label><br><br>
            <label>As previsões desta IA ainda não podem ser consideradas como certas, ela ainda está em desenvolvimento.</label><br><br>
            <label><b>Previsão com base no histórico dos dois times:</b></label>
        </div>
        <div class="div-estatisticas-team">
            <div class="div-estatisticas-team-winner">
                <label>Vitória: ${probsIA["previsao_home"]["vitoria"]}</label><br>
            </div>
            <div class="div-estatisticas-team-empate">
                <label>Empate: ${probsIA["previsao_home"]["empate"]}</label><br>
            </div>
            <div class="div-estatisticas-team-derrota">
                <label>Derrota: ${probsIA["previsao_home"]["derrota"]}</label><br>
            </div>
        </div>
    `

    div_estatisticas_team_away.innerHTML = `
        <div class="div-info-resultados-ia">
            <label>Previsões geradas pela IA versão: ${probsIA["v_ia"]} </label><br>
            <label>Erro da rede ficou em: ${probsIA["erro"]} </label><br>
            <label>Foi usado os ultimos ${probsIA["qtde_dados"]} jogos desse time. </label><br>
            <label>Previsão pra o jogo do dia: ${probsIA["data_jogo_previsto"]}</label><br><br>
            <label>As previsões desta IA ainda não podem ser consideradas como certas, ela ainda está em desenvolvimento.</label><br><br>
            <label><b>Previsão com base no histórico dos dois times:</b></label>
        </div>
        <div class="div-estatisticas-team">
            <div class="div-estatisticas-team-winner">
                <label>Vitória: ${probsIA["previsao_away"]["vitoria"]}</label><br>
            </div>
            <div class="div-estatisticas-team-empate">
                <label>Empate: ${probsIA["previsao_away"]["empate"]}</label><br>
            </div>
            <div class="div-estatisticas-team-derrota">
                <label>Derrota: ${probsIA["previsao_away"]["derrota"]}</label><br>
            </div>
        </div>
    `
}

async function fazerRequisicaoParaIAPreverTime(name_id_div, id_season, id_team, is_home){
    let div_previsao_team = document.getElementById(name_id_div);
    let params = "/previsao-team?id_season="+id_season+"&id_team="+id_team
    let probsIA = await callGETAPI(params, true, true)

    div_previsao_team.innerHTML = `
        <div class="div-info-resultados-ia">
            <label>Previsões geradas pela IA versão: ${probsIA["v_ia"]} </label><br>
            <label>Erro da rede ficou em: ${probsIA["erro"]} </label><br>
            <label>Foi usado os ultimos ${probsIA[`qtde_dados`]} jogos desse time. </label><br>
            <label>Previsão pra o jogo do dia: ${probsIA["data_jogo_previsto"]}</label><br><br>
            <label>As previsões desta IA ainda não podem ser consideradas como certas, ela ainda está em desenvolvimento.</label><br><br>
            <label><b>Previsão com base no histórico desse único time:</b></label>
        </div>
        <div class="div-estatisticas-team">
            <div class="div-estatisticas-team-winner">
                <label>Vitória: ${probsIA["previsao"]["vitoria"]}</label><br>
            </div>
            <div class="div-estatisticas-team-empate">
                <label>Empate: ${probsIA["previsao"]["empate"]}</label><br>
            </div>
            <div class="div-estatisticas-team-derrota">
                <label>Derrota: ${probsIA["previsao"]["derrota"]}</label><br>
            </div>
        </div>
    `
}

async function fazerRequisicaoEstatisticas(name_id_div, id_season, id_team, is_home){
    let params = "/statistics?id_season="+id_season+"&id_team="+id_team
    let arrTeamsEstatisticas = await callGETAPI(params, true, true)
    let div_estatisticas = document.getElementById(name_id_div)
    div_estatisticas.classList.add("grid-2-colunas")
    div_estatisticas.classList.add("margin-horizontal-15px")
    div_estatisticas.classList.add("margin-vertical-15px")

    console.log(arrTeamsEstatisticas)

    if(arrTeamsEstatisticas.length >= 2){
        alert("eitaa BB, retornando mais doque devia para as Estatisticas")
        throw new Error("eitaa BB, retornando mais doque devia para as Estatisticas")
    }else if(arrTeamsEstatisticas.length === 0){
        return
    }


    let teamStatistics = arrTeamsEstatisticas[0]
    let arrTeamEstatisticas = teamStatistics["arr_dataset_data_fixture_team"].slice(-1)
    let arrMediasTeamEstatisticas = arrTeamEstatisticas[0]["media_estatisticas"]
    console.log(arrMediasTeamEstatisticas)

    div_estatisticas.innerHTML = `
        <div class="display-flex-row-space-between backgroud-color-444448 width-100 align-items-center font-weight-bold">
            <label class="text-align-center width-100 color-white">
                Estátisticas para o dia: ${arrTeamEstatisticas[0]["data_fixture"]}
            </label>
        </div>
    `

    for(let media of arrMediasTeamEstatisticas){
        let nameClassColorGoodEstatistica = obterNameClassColorGoodEstatistica(media)
        let setaInclinacao = obterSetaInclinacaoMedia(media)
        div_estatisticas.innerHTML += `
            <div class="display-flex-row-space-between backgroud-color-444448">
                <label class="paddaing-horizontal-15px paddaing-vertical-5px width-80 color-white">
                    ${media["name_statistic"]}: ${setaInclinacao}
                </label>
                <label class="width-20 paddaing-vertical-5px color-black 
                              ${nameClassColorGoodEstatistica} text-align-center">
                    ${media["media_"+media["id_statistic"]+ "_formatada"]} 
                </label>
            </div>
            
        `
    }
}

function ajustarLayout(){
    if (/Mobi/.test(navigator.userAgent) || document.documentElement.clientWidth <= 600){
        let div_menu_left = document.getElementById("div-menu-left")
        let div_team_x_team = document.getElementById("div-team-x-team")
        let div_team_x_home = document.getElementById("div-team-x-home")
        let div_team_x_away = document.getElementById("div-team-x-away")


        div_menu_left.style.width = "40vw"

        div_team_x_team.classList.remove("team-x-team")
        div_team_x_team.classList.add("team-x-team-mobile")

        div_team_x_home.classList.remove("team-x")
        div_team_x_home.classList.add("team-x-mobile")

        div_team_x_away.classList.remove("team-x")
        div_team_x_away.classList.add("team-x-mobile")
    }
}

function setImgTeam(isHome, logo_team){
    let imgTeamSelected = document.getElementById("img-team" + (isHome ? "-home" : "-away"))
    imgTeamSelected.setAttribute("src", logo_team)
    imgTeamSelected.setAttribute("width", document.documentElement.clientWidth * 0.33)
}

function obterNameClassColorGoodEstatistica(mediaEstatistica){
    let is_caindo = mediaEstatistica["is_caindo"]
    let is_decline_good = mediaEstatistica["is_decline_good"]
    let inclinacao_media = mediaEstatistica["inclinacao_media"]
    let media = mediaEstatistica["media_"+mediaEstatistica["id_statistic"]]
    let name_class = "backgroud-color-empate"

    //0.005 == 0,5%
    if(inclinacao_media < 0.005 || media === 0){
        return name_class
    }

    if(is_caindo){
        if(is_decline_good){
            name_class = "backgroud-color-vitoria"
        }else{
            name_class = "backgroud-color-derrota"
        }
    }else{
        if(is_decline_good){
            name_class = "backgroud-color-derrota"
        }else{
            name_class = "backgroud-color-vitoria"
        }

    }

    return name_class
}

function obterSetaInclinacaoMedia(mediaEstatistica){
    let is_caindo = mediaEstatistica["is_caindo"]
    let is_decline_good = mediaEstatistica["is_decline_good"]
    let inclinacao_media = mediaEstatistica["inclinacao_media"]
    let name_class = `<i class="fa-solid fa-grip-lines color-empate float-right"></i>`
    let media = mediaEstatistica["media_"+mediaEstatistica["id_statistic"]]

    //0.005 == 0,5%
    if(inclinacao_media < 0.005 || media === 0){
        return name_class
    }

    if(is_caindo){
        if(is_decline_good){
            name_class = `<i class="fa-solid fa-arrow-down color-vitoria float-right"></i>`
        }else{
            name_class = `<i class="fa-solid fa-arrow-down color-derrota float-right"></i>`
        }
    }else{
        if(is_decline_good){
            name_class = `<i class="fa-solid fa-arrow-up color-derrota float-right"></i>`
        }else {
            name_class = `<i class="fa-solid fa-arrow-up color-vitoria float-right"></i>`
        }

    }

    return name_class
}

