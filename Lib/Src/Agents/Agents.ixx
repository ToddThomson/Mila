/**
 * @file Agents.ixx
 * @brief Mila::Agents, the library's agents API: a model at work in a program, with its session, its tools and the
 * loop that runs their calls (Direction.md 3.2, ToolCalling.md).
 */

export module Mila.Agents;

export import Agents.ToolNeeds;
export import Agents.FolderAccess;
export import Agents.GrantFolder;
export import Agents.FolderEntry;
export import Agents.GrantPath;
export import Agents.PublicAddress;
export import Agents.GrantFile;
export import Agents.ProgramRequest;
export import Agents.ProgramResult;
export import Agents.ProgramFolders;
export import Agents.ContainedRun;
export import Agents.ProgramContainer;
export import Agents.MxcProgramContainer;
export import Agents.ToolGrant;
export import Agents.Tool;
export import Agents.FileTools;
export import Agents.HtmlText;
export import Agents.WebPageReader;
export import Agents.ToolQuestion;
export import Agents.ToolAnswer;
export import Agents.ToolPermission;
export import Agents.ArgumentCondition;
export import Agents.ToolRule;
export import Agents.ToolRules;
export import Agents.ToolDecision;
export import Agents.ToolOutcome;
export import Agents.ArgumentSchema;
export import Agents.ToolGate;
export import Agents.AgentEvent;
export import Agents.AgentLimits;
export import Agents.Response;
export import Agents.SamplingOverride;
export import Agents.Session;
export import Agents.Agent;
