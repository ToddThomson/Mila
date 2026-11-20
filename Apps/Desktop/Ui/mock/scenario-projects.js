// Scenario 16: a workspace with projects (AssistantInterface.md phase 4). Three projects hold some of the seeded chats:
// Calculator app with its folder to read and change and a plan stopped for the person, Kyoto in November with none,
// and Local LLMs on 16 GB with two folders to read. A turn is scenario 14's, Mila asking for a folder, so in Local
// LLMs the question offers the project's two folders before the picker.

import { folderRequest } from './scenario-folder-request.js';

export const projectsWorkspace = {
    ...folderRequest,
    number: 16,
    title: 'A workspace with projects',
    projects: true
};
