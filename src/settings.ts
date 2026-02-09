import powerbi from "powerbi-visuals-api";

export const chatbotSettingsObjectName = "chatbotSettings";
export const endpointUrlPropertyName = "endpointUrl";

export class ChatbotSettings {
  public endpointUrl = "";

  public static parse(dataView?: powerbi.DataView): ChatbotSettings {
    const settings = new ChatbotSettings();
    const objects = dataView?.metadata?.objects;
    settings.endpointUrl = getValue(objects, chatbotSettingsObjectName, endpointUrlPropertyName, "");
    return settings;
  }
}

export function buildFormattingModel(settings: ChatbotSettings): powerbi.visuals.FormattingModel {
  return {
    cards: [
      {
        uid: `${chatbotSettingsObjectName}_card`,
        displayName: "Chatbot",
        groups: [
          {
            uid: `${chatbotSettingsObjectName}_group`,
            displayName: "Endpoint",
            slices: [
              {
                uid: `${chatbotSettingsObjectName}_endpointUrl`,
                displayName: "Endpoint URL",
                control: {
                  type: powerbi.visuals.FormattingComponent.TextInput,
                  properties: {
                    placeholder: "http://localhost:8000/chat",
                    descriptor: {
                      objectName: chatbotSettingsObjectName,
                      propertyName: endpointUrlPropertyName
                    },
                    value: settings.endpointUrl
                  }
                }
              }
            ]
          }
        ]
      }
    ]
  };
}

function getValue<T>(
  objects: powerbi.DataViewObjects | undefined,
  objectName: string,
  propertyName: string,
  defaultValue: T
): T {
  if (!objects) {
    return defaultValue;
  }

  const object = objects[objectName] as powerbi.DataViewObject | undefined;
  if (!object) {
    return defaultValue;
  }

  const value = object[propertyName];
  if (value === undefined || value === null) {
    return defaultValue;
  }

  return value as T;
}
