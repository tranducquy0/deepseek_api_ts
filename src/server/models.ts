import { Router } from "express";
import { buildModelList } from "../shared/convert.js";
import type { OpenAIModelList } from "../shared/types.js";

const MODELS_RESPONSE: OpenAIModelList = buildModelList();

export function modelsRouter(): Router {
  const router = Router();

  router.get("/v1/models", (_req, res) => {
    res.json(MODELS_RESPONSE);
  });

  return router;
}
