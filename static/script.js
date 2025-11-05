async function sendMessage() {
  const input = document.getElementById("user-input");
  const chatBox = document.getElementById("chat-box");
  const message = input.value.trim();
  if (!message) return;

  chatBox.innerHTML += `<div class="message user"><b>You:</b> ${message}</div>`;
  input.value = "";

  const response = await fetch("/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ message }),
  });

  const data = await response.json();
  chatBox.innerHTML += `<div class="message bot"><b>Bot:</b> ${data.response}</div>`;
  chatBox.scrollTop = chatBox.scrollHeight;
}
