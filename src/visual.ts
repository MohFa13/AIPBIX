import powerbi from "powerbi-visuals-api";
import { buildFormattingModel, ChatbotSettings } from "./settings";

import IVisual = powerbi.extensibility.visual.IVisual;
import VisualConstructorOptions = powerbi.extensibility.visual.VisualConstructorOptions;
import VisualUpdateOptions = powerbi.extensibility.visual.VisualUpdateOptions;

export default class ChatbotVisual implements IVisual {
  private root: HTMLDivElement;
  private chatHistory: HTMLDivElement;
  private inputBox: HTMLInputElement;
  private sendButton: HTMLButtonElement;
  private settings: ChatbotSettings;

  constructor(options: VisualConstructorOptions) {
    this.settings = new ChatbotSettings();

    this.root = document.createElement("div");
    this.root.className = "chatbot-visual";

    this.chatHistory = document.createElement("div");
    this.chatHistory.className = "chatbot-visual__history";

    const inputRow = document.createElement("div");
    inputRow.className = "chatbot-visual__input-row";

    this.inputBox = document.createElement("input");
    this.inputBox.type = "text";
    this.inputBox.className = "chatbot-visual__input";
    this.inputBox.placeholder = "Type your message...";

    this.sendButton = document.createElement("button");
    this.sendButton.type = "button";
    this.sendButton.className = "chatbot-visual__send";
    this.sendButton.textContent = "Send";

    inputRow.appendChild(this.inputBox);
    inputRow.appendChild(this.sendButton);

    this.root.appendChild(this.chatHistory);
    this.root.appendChild(inputRow);

    options.element.appendChild(this.root);

    this.sendButton.addEventListener("click", () => {
      void this.handleSend();
    });

    this.inputBox.addEventListener("keydown", (event: KeyboardEvent) => {
      if (event.key === "Enter") {
        event.preventDefault();
        void this.handleSend();
      }
    });
  }

  public update(options: VisualUpdateOptions): void {
    this.settings = ChatbotSettings.parse(options.dataViews?.[0]);

    const viewport = options.viewport;
    if (viewport) {
      this.root.style.width = `${viewport.width}px`;
      this.root.style.height = `${viewport.height}px`;
    }
  }

  public getFormattingModel(): powerbi.visuals.FormattingModel {
    return buildFormattingModel(this.settings);
  }

  private async handleSend(): Promise<void> {
    const message = this.inputBox.value.trim();
    if (!message) {
      return;
    }

    this.appendMessage(message, true);
    this.inputBox.value = "";

    if (!this.settings.endpointUrl) {
      this.appendMessage("Endpoint URL is not configured.", false);
      return;
    }

    try {
      const response = await fetch(this.settings.endpointUrl, {
        method: "POST",
        headers: {
          "Content-Type": "application/json"
        },
        body: JSON.stringify({ message })
      });

      if (!response.ok) {
        throw new Error(`Status ${response.status}`);
      }

      const contentType = response.headers.get("Content-Type") ?? "";
      let replyText = "";

      if (contentType.includes("application/json")) {
        const data = (await response.json()) as Record<string, unknown>;
        const candidate =
          (typeof data.response === "string" && data.response) ||
          (typeof data.message === "string" && data.message) ||
          (typeof data.text === "string" && data.text) ||
          "";
        replyText = candidate || JSON.stringify(data);
      } else {
        replyText = await response.text();
      }

      if (!replyText) {
        replyText = "No response received from the chatbot.";
      }

      this.appendMessage(replyText, false);
    } catch (error) {
      this.appendMessage("Unable to reach the chatbot endpoint.", false);
    }
  }

  private appendMessage(text: string, isUser: boolean): void {
    const messageElement = document.createElement("div");
    messageElement.className = `chatbot-visual__message ${
      isUser ? "chatbot-visual__message--user" : "chatbot-visual__message--bot"
    }`;
    messageElement.textContent = text;

    this.chatHistory.appendChild(messageElement);
    this.chatHistory.scrollTop = this.chatHistory.scrollHeight;
  }
}
