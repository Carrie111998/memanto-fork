import { defineTool } from "eve/tools";
import { z } from "zod";

import type { Memanto } from "../index.js";
import { MEMORY_TYPES, type MemantoToolName, type MemoryType } from "./memory-types.js";

export { MEMORY_TYPES };
export type { MemantoToolName, MemoryType };

export interface CreateMemantoEveToolsOptions {
  /**
   * Which tools to create. Defaults to all of them. Pass a subset to expose
   * only, say, read access: `{ include: ["recallMemory"] }`.
   */
  include?: MemantoToolName[];
  /**
   * Default result limit applied to `recallMemory` / `answerMemory` when the
   * model does not specify one. Falls back to the server default when omitted.
   */
  defaultLimit?: number;
}

/**
 * Build eve tools backed by a {@link Memanto} client.
 *
 * eve discovers one tool per file under `agent/tools/`, so re-export a
 * single tool from each file rather than the whole map:
 *
 * ```ts
 * // agent/tools/recall-memory.ts
 * import { Memanto } from "@moorcheh-ai/memanto";
 * import { createMemantoEveTools } from "@moorcheh-ai/memanto/eve";
 *
 * const memanto = new Memanto({ agentId: "my-agent" });
 *
 * export default createMemantoEveTools(memanto).recallMemory;
 * ```
 *
 * Repeat for `agent/tools/remember-memory.ts` (`.rememberMemory`) and
 * `agent/tools/answer-memory.ts` (`.answerMemory`).
 *
 * `eve` and `zod` are optional peer dependencies — install them in the host
 * project (eve projects already depend on both).
 */
export function createMemantoEveTools(
  memanto: Memanto,
  options: CreateMemantoEveToolsOptions = {},
) {
  const { include, defaultLimit } = options;

  const all = {
    recallMemory: defineTool({
      description:
        "Search the user's long-term memory for relevant facts, preferences, " +
        "decisions, or past context. Call this before answering whenever the " +
        "user refers to information from earlier or from a previous session.",
      inputSchema: z.object({
        query: z
          .string()
          .min(1)
          .describe("Natural-language description of what to recall"),
        limit: z
          .number()
          .int()
          .min(1)
          .max(50)
          .optional()
          .describe("Maximum number of memories to return"),
        type: z
          .array(z.enum(MEMORY_TYPES))
          .optional()
          .describe("Optional filter restricting results to these memory types"),
      }),
      execute: async ({ query, limit, type }: { query: string; limit?: number; type?: MemoryType[] }) => {
        const res = (await memanto.recall({
          query,
          limit: limit ?? defaultLimit,
          type,
        })) as { memories?: unknown };
        return res.memories ?? res;
      },
    }),

    rememberMemory: defineTool({
      description:
        "Persist a durable fact, preference, decision, or instruction that " +
        "will be useful in future sessions. Do not store secrets, credentials, " +
        "or transient chatter.",
      inputSchema: z.object({
        content: z.string().min(1).describe("The information to remember"),
        type: z
          .enum(MEMORY_TYPES)
          .optional()
          .describe("Memory type. Omit to let the server auto-classify."),
        title: z.string().optional().describe("Optional short title"),
        tags: z
          .array(z.string())
          .optional()
          .describe("Optional tags for later filtering"),
      }),
      execute: async ({
        content,
        type,
        title,
        tags,
      }: {
        content: string;
        type?: MemoryType;
        title?: string;
        tags?: string[];
      }) => memanto.remember({ content, type, title, tags }),
    }),

    answerMemory: defineTool({
      description:
        "Answer a question using retrieval-augmented generation over the " +
        "user's stored memories. Prefer this over recallMemory when a direct, " +
        "synthesized answer from memory is more useful than raw results.",
      inputSchema: z.object({
        question: z
          .string()
          .min(1)
          .describe("The question to answer from memory"),
        limit: z
          .number()
          .int()
          .min(1)
          .max(100)
          .optional()
          .describe("Number of context memories to use"),
      }),
      execute: async ({ question, limit }: { question: string; limit?: number }) =>
        memanto.answer({ question, limit: limit ?? defaultLimit }),
    }),
  };

  if (!include) return all;

  const selected = {} as Partial<typeof all>;
  for (const name of Object.keys(all) as MemantoToolName[]) {
    if (include.includes(name)) {
      (selected as Record<MemantoToolName, (typeof all)[MemantoToolName]>)[name] =
        all[name];
    }
  }
  return selected;
}
