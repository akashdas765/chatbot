const popup = document.querySelector('.chat-popup');
const chatBtn = document.querySelector('.chat-btn');
const submitBtn = document.querySelector('.submit');
const chatArea = document.querySelector('.chat-area');
const inputElm = document.querySelector('input');

//   chat button toggler

chatBtn.addEventListener('click', ()=>{
    popup.classList.toggle('show');
})

function response(text)
{
  $.get("/get", { msg: text }).done(function(data) {
    var u = String(data);
    var z=u.split(" ");
    var t = z[0];
    //delete z[0];
    z=z.splice(1);
    var y = z.join(" ");
    t=parseInt(t);
    var botHtml = '<p class="botText"><span>' + y + "</span></p>";
    $("#chatarea").append(botHtml);
    switch(t)
    {
      case 6:
        res1();
        break;
      case 8:
        res2();
        break;
      case 15:
        res2();
        break;
      default:
        break;
    }
    document
      .getElementById("userInput")
      .scrollIntoView({ block: "start", behavior: "smooth" });
  });
}


submitBtn.addEventListener('click', ()=>{
    let userInput = inputElm.value;

    let temp = `<div class="out-msg">
    <span class="my-msg">${userInput}</span>
    <img src="img/me.jpg" class="avatar">
    </div>`;

    chatArea.insertAdjacentHTML("beforeend", temp);
    response(userInput);

    inputElm.value = '';

})
