"use strict";
localStorage.country = "GB";
localStorage.lang = "en";
localStorage.id = "4f41763c-65af-11ec-a36f-0242ac160002";
function load_tags() {
    const xhttp = new XMLHttpRequest();
    xhttp.onload = function() {
        localStorage.all_tags = this.responseText;
        top_tags('');
        sample();
    }
    xhttp.open("GET", `/all-tags?lang=${localStorage.lang}`);
    xhttp.send();
}
function render_table(models, el, ex) {
    let names = [];
    models["schema"]["fields"].forEach((field)=>  names.push(field["name"]) );
    let thead = document.createElement("thead");
    let tr = document.createElement("tr");
    names.filter(name=> !ex.includes(name)).forEach((name)=> {
        let th = document.createElement("th"); 
        th.innerHTML = name; 
        tr.appendChild(th);
    });
    thead.appendChild(tr);
    let tbody = document.createElement("tbody");
    models["data"].forEach((model)=>{
        let tr = document.createElement("tr");
        names.filter(name=> !ex.includes(name)).forEach((name) => {
            let td = document.createElement("td");
            td.innerHTML = model[name];
            tr.appendChild(td); 
            tr.onclick = () => {
                document.querySelectorAll('.table-active').forEach( (e)=> e.classList.remove("table-active") );
                tr.classList.add("table-active");
                if (model.id != localStorage.id) {
                    localStorage.id = model.id;
                    localStorage.lang = model.lang;
                    localStorage.country = ( (model.lang=='en') ? "GB" : model.lang.toUpperCase() );
                    load_tags();
                }
            }
        });
        tbody.appendChild(tr);
    });
    let table = document.createElement("table");
    table.classList.add("table");
    table.classList.add("table-responsive");
    table.appendChild(thead);
    table.appendChild(tbody);
    el.replaceChildren(table);
}
function load_models() {
    const xhttp = new XMLHttpRequest();
    xhttp.onload = function() {
        let models = JSON.parse(this.responseText);
        let div_element = document.getElementById("models")
        render_table(models, div_element, ["id"]); // "model_name","case_insensitive","id","ngram_min","title_alt_imp"]);
    }
    xhttp.open("GET", "/all-models");
    xhttp.send();
}
function sample() {
    const xhttp = new XMLHttpRequest();
    xhttp.onload = function() {
        let response = JSON.parse(this.responseText);
        document.getElementById("job-title").innerHTML = response['job_title'];
        document.getElementById("job-description").innerHTML = response['job_description'];
        top_tags('');
    }
    xhttp.open("GET", `/sample-vacancy?country=${localStorage.country}`);
    xhttp.send();
}
function top_tags(text) {
    var all_tags = JSON.parse(localStorage.all_tags);
    const xhttp = new XMLHttpRequest();
    xhttp.onload = function() {
        let response = JSON.parse(this.responseText);
        document.getElementById("tags").replaceChildren();        
        response.slice(0,10).forEach((top_occ)=>{
            let occupation_id = all_tags.data.map((occ)=>occ.occupation_id);
            let occ = all_tags.data[occupation_id.indexOf(top_occ.index)]
            let tag = document.createElement("button"); 
            tag.classList.add("btn");
            tag.classList.add("m-1");
            tag.classList.add("rounded-pill");
            tag.classList.add("btn-secondary");
            tag.setAttribute("data-bs-toggle","tooltip");
            tag.setAttribute("title",occ.description);
            tag.innerText = occ.title;
            document.getElementById("tags").appendChild(tag);
        });
        let tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
        let tooltipList = tooltipTriggerList.map(function (tooltipTriggerEl) {
            return new bootstrap.Tooltip(tooltipTriggerEl);
        })
        }
    xhttp.open("POST", "/top-tags");
    xhttp.setRequestHeader("Content-Type", "application/json")
    if (text==null || text.trim().length==0) 
    {
        var request = {
            id: localStorage.id.toString(), 
            title: document.getElementById("job-title").innerHTML, 
            description: document.getElementById("job-description").innerHTML 
        }
        } else {
            var request = {id: localStorage.id.toString(), description: text}
            }

    xhttp.send(JSON.stringify(request));
}
load_models();
load_tags();
sample();