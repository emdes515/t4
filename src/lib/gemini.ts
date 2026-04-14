import { GoogleGenAI, Type } from "@google/genai";
import { z } from "zod";

export const JobSchema = z.object({
  is_valid_job_offer: z.boolean(),
  confidence_score: z.number().int().min(0).max(100).nullable().optional(),
  job_title: z.string().nullable().optional(),
  seniority_level: z.enum(['junior', 'mid', 'senior', 'lead', 'unknown']).optional().default('unknown'),
  company_name: z.string().nullable().optional(),
  location: z.string().nullable().optional(),
  salary: z.string().nullable().optional(),
  contract_type: z.string().nullable().optional(),
  work_model: z.string().nullable().optional(),
  working_hours: z.string().nullable().optional(),
  skills: z.preprocess((val) => {
    if (Array.isArray(val)) {
      return { must_have_hard: val, nice_to_have_hard: [], soft_skills: [] };
    }
    return val;
  }, z.object({
    must_have_hard: z.array(z.string()).optional().default([]),
    nice_to_have_hard: z.array(z.string()).optional().default([]),
    soft_skills: z.array(z.string()).optional().default([])
  }).optional().default({ must_have_hard: [], nice_to_have_hard: [], soft_skills: [] })),
  job_context: z.object({
    daily_tasks: z.array(z.string()).optional().default([]),
    pain_points: z.string().nullable().optional(),
    benefits: z.array(z.string()).optional().default([])
  }).optional().default({ daily_tasks: [], pain_points: null, benefits: [] }),
  ats_keywords: z.array(z.string()).optional().default([])
});

export type JobInfo = z.infer<typeof JobSchema>;

export const getSystemInstruction = (userLanguage: string) => `
You are a strict, highly analytical HR Data Extraction Engine. 
Visit and analyze the provided job offer URL or text. Your ONLY task is to extract factual information and return it strictly as JSON.

CRITICAL RULES:
1. IGNORE NOISE: Ignore sections like 'Similar jobs', 'Recommended for you', website menus, and footers. Read ONLY the main job description.
2. NO HALLUCINATIONS: If a specific detail (e.g., salary, contract type) is NOT explicitly stated in the text, you MUST return null. Do not guess.
3. VALIDITY CHECK: If the page is a 404 error, a captcha challenge, a login screen, or completely unrelated to a job offer, set "is_valid_job_offer" to false and leave the rest empty.
4. OUTPUT LANGUAGE: You must translate or formulate ALL extracted text values (job title, skills, tasks, benefits, etc.) into the following language: ${userLanguage}.

Expected JSON Output format:
{
  "is_valid_job_offer": boolean,
  "confidence_score": "Integer 0-100. How well did you understand and parse this job offer? Return null if uncertain.",
  "job_title": string | null,
  "seniority_level": "Infer from context, salary, and requirements. Return one of: junior | mid | senior | lead | unknown",
  "company_name": string | null,
  "location": string | null,
  "salary": string | null,
  "contract_type": string | null,
  "work_model": string | null,
  "working_hours": string | null,
  "skills": {
    "must_have_hard": [string],
    "nice_to_have_hard": [string],
    "soft_skills": [string]
  },
  "job_context": {
    "daily_tasks": [string],
    "pain_points": string | null,
    "benefits": [string]
  },
  "ats_keywords": "Array of 5-12 most important ATS keywords extracted from the job offer. Focus on hard skills, technologies, tools, and role-specific terminology."
}
`;

const responseSchema = {
  type: Type.OBJECT,
  properties: {
    is_valid_job_offer: { type: Type.BOOLEAN, description: "false if the page is just a captcha, login page, or error 404" },
    confidence_score: { type: Type.INTEGER, description: "Integer 0-100. How well did you understand and parse this job offer? Return null if uncertain.", nullable: true },
    job_title: { type: Type.STRING, description: "The exact job title.", nullable: true },
    seniority_level: { type: Type.STRING, description: "Infer from context, salary, and requirements. Return one of: junior | mid | senior | lead | unknown", nullable: true },
    company_name: { type: Type.STRING, description: "The exact company name.", nullable: true },
    location: { type: Type.STRING, description: "Exact location or 'Remote'", nullable: true },
    salary: { type: Type.STRING, description: "Salary range if explicitly stated, e.g., '5000-7000 PLN brutto', '35 PLN/h', else null.", nullable: true },
    contract_type: { type: Type.STRING, description: "Type of contract, e.g., 'Umowa o pracę', 'B2B', else null.", nullable: true },
    work_model: { type: Type.STRING, description: "Work model, e.g., 'Stacjonarnie', 'Hybrydowo', 'Zdalnie', else null.", nullable: true },
    working_hours: { type: Type.STRING, description: "Working hours, e.g., 'Praca 3-zmianowa', '9:00 - 17:00', 'Nocki', else null.", nullable: true },
    skills: {
      type: Type.OBJECT,
      properties: {
        must_have_hard: { type: Type.ARRAY, items: { type: Type.STRING }, description: "Extract ONLY explicitly listed mandatory hard requirements/tools" },
        nice_to_have_hard: { type: Type.ARRAY, items: { type: Type.STRING }, description: "Extract ONLY explicitly listed optional/bonus hard skills" },
        soft_skills: { type: Type.ARRAY, items: { type: Type.STRING }, description: "Extract explicitly listed soft skills" }
      }
    },
    job_context: {
      type: Type.OBJECT,
      properties: {
        daily_tasks: { type: Type.ARRAY, items: { type: Type.STRING }, description: "List core duties explicitly written in the text" },
        pain_points: { type: Type.STRING, description: "What problem is the company trying to solve by hiring this role?", nullable: true },
        benefits: { type: Type.ARRAY, items: { type: Type.STRING }, description: "List of benefits offered (e.g., 'Multisport', 'Remote work')." }
      }
    },
    ats_keywords: { type: Type.ARRAY, items: { type: Type.STRING }, description: "Array of 5-12 most important ATS keywords extracted from the job offer. Focus on hard skills, technologies, tools, and role-specific terminology." }
  },
  required: ["is_valid_job_offer"]
};

export class ScraperBlockedError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'ScraperBlockedError';
  }
}

export class InvalidJobOfferError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'InvalidJobOfferError';
  }
}

export const fetchJobFromURL = async (apiKey: string, url: string, userLanguage: string) => {
  const ai = new GoogleGenAI({ apiKey });

  const fetchWithGemini3Flash = async () => {
    const response = await ai.models.generateContent({
      model: "gemini-3-flash",
      contents: `Analyze the job offer available at this URL: ${url}
      
Your task is to analyze the content found at this link. Use your internet access capabilities (e.g., Google Search) to read the job posting.
Extract all details into a structured JSON format. Pay attention to hidden requirements, technologies, and responsibilities.

Return ONLY a clean JSON object, without markdown tags (no \`\`\`json).

If you cannot access the page or the link does not contain a job offer, return a JSON with \`is_valid_job_offer: false\`.`,
      config: {
        temperature: 0.0,
        systemInstruction: getSystemInstruction(userLanguage),
        tools: [{ googleSearch: {} }],
        responseMimeType: "application/json",
        responseSchema
      }
    });

    if (response.usageMetadata) {
      console.log('[Gemini Usage] fetchJobFromURL', {
        model: 'gemini-3-flash',
        promptTokens: response.usageMetadata.promptTokenCount,
        candidateTokens: response.usageMetadata.candidatesTokenCount,
        totalTokens: response.usageMetadata.totalTokenCount,
        timestamp: new Date().toISOString(),
      });
    }

    return response.text;
  };

  const fetchWithTwoStagePipeline = async () => {
    const searchResponse = await ai.models.generateContent({
      model: "gemini-2.5-flash",
      contents: `Fetch the content of this job offer URL: ${url}. Return the full text of the job description.`,
      config: {
        temperature: 0.0,
        tools: [{ googleSearch: {} }]
      }
    });
    
    const scrapedText = searchResponse.text;
    
    const analyzeResponse = await ai.models.generateContent({
      model: "gemini-2.5-flash",
      contents: `Analyze the following job description:\n\n${scrapedText}`,
      config: {
        temperature: 0.0,
        systemInstruction: getSystemInstruction(userLanguage),
        responseMimeType: "application/json",
        responseSchema
      }
    });
    
    if (analyzeResponse.usageMetadata) {
      console.log('[Gemini Usage] fetchJobFromURL (Two-Stage)', {
        model: 'gemini-2.5-flash',
        promptTokens: analyzeResponse.usageMetadata.promptTokenCount,
        candidateTokens: analyzeResponse.usageMetadata.candidatesTokenCount,
        totalTokens: analyzeResponse.usageMetadata.totalTokenCount,
        timestamp: new Date().toISOString(),
      });
    }
    
    return analyzeResponse.text;
  };

  let responseText: string | undefined;

  try {
    responseText = await fetchWithGemini3Flash();
  } catch (error: any) {
    if (error?.status === 404 || error?.message?.includes('not found') || error?.message?.includes('INVALID_ARGUMENT')) {
      console.warn('[Gemini] gemini-3-flash unavailable or failed, falling back to two-stage pipeline');
      responseText = await fetchWithTwoStagePipeline();
    } else {
      throw error;
    }
  }

  if (!responseText) {
    throw new ScraperBlockedError("Model AI nie zwrócił żadnych danych. Proszę, skopiuj treść ogłoszenia i wklej ją ręcznie.");
  }

  try {
    const parsed = JSON.parse(responseText);
    const validated = JobSchema.parse(parsed);
    
    if (!validated.is_valid_job_offer) {
       throw new InvalidJobOfferError("Podany URL nie prowadzi do ogłoszenia o pracę. Sprawdź link lub wklej treść ogłoszenia ręcznie.");
    }
    
    return validated;
  } catch (e) {
    if (e instanceof ScraperBlockedError || e instanceof InvalidJobOfferError) throw e;
    
    responseText = responseText.replace(/```json\n?/g, '').replace(/```\n?/g, '').trim();
    try {
      const parsed = JSON.parse(responseText);
      const validated = JobSchema.parse(parsed);
      
      if (!validated.is_valid_job_offer) {
         throw new InvalidJobOfferError("Podany URL nie prowadzi do ogłoszenia o pracę. Sprawdź link lub wklej treść ogłoszenia ręcznie.");
      }
      return validated;
    } catch (e2) {
      if (e2 instanceof ScraperBlockedError || e2 instanceof InvalidJobOfferError) throw e2;
      console.error("Zod validation or JSON parsing failed:", e2);
      throw new ScraperBlockedError("Błąd przetwarzania danych z AI. Proszę, skopiuj treść ogłoszenia i wklej ją ręcznie.");
    }
  }
};

export const analyzeJobDescription = async (apiKey: string, text: string, userLanguage: string) => {
  const ai = new GoogleGenAI({ apiKey });
  const response = await ai.models.generateContent({
    model: "gemini-3-flash",
    contents: `Analyze the following job description:\n\n${text}`,
    config: {
      temperature: 0.0,
      systemInstruction: getSystemInstruction(userLanguage),
      responseMimeType: "application/json",
      responseSchema
    }
  });

  if (response.usageMetadata) {
    console.log('[Gemini Usage]', {
      model: 'gemini-3-flash',
      promptTokens: response.usageMetadata.promptTokenCount,
      candidateTokens: response.usageMetadata.candidatesTokenCount,
      totalTokens: response.usageMetadata.totalTokenCount,
      timestamp: new Date().toISOString(),
    });
  }

  let responseText = response.text;

  if (!responseText) {
    throw new InvalidJobOfferError("Podany tekst nie wygląda na ogłoszenie o pracę. Sprawdź treść i spróbuj ponownie.");
  }

  try {
    const parsed = JSON.parse(responseText);
    const validated = JobSchema.parse(parsed);
    if (!validated.is_valid_job_offer) {
      throw new InvalidJobOfferError("Podany tekst nie wygląda na ogłoszenie o pracę. Sprawdź treść i spróbuj ponownie.");
    }
    return validated;
  } catch (e) {
    if (e instanceof InvalidJobOfferError) throw e;
    responseText = responseText.replace(/```json\n?/g, '').replace(/```\n?/g, '').trim();
    try {
      const parsed = JSON.parse(responseText);
      const validated = JobSchema.parse(parsed);
      if (!validated.is_valid_job_offer) {
        throw new InvalidJobOfferError("Podany tekst nie wygląda na ogłoszenie o pracę. Sprawdź treść i spróbuj ponownie.");
      }
      return validated;
    } catch (e2) {
      if (e2 instanceof InvalidJobOfferError) throw e2;
      console.error("Zod validation or JSON parsing failed:", e2);
      throw new Error("Błąd przetwarzania danych z AI. Spróbuj ponownie.");
    }
  }
};

export const enhanceText = async (apiKey: string, text: string, context: string = "") => {
  const ai = new GoogleGenAI({ apiKey });
  const prompt = `
    Analyze the following text provided by a user for their CV/Resume.
    
    TEXT: "${text}"
    CONTEXT: "${context}"
    
    INSTRUCTIONS:
    1. VALIDATION: Check if the text makes sense and contains enough context to be improved. If it's gibberish (e.g., "asdsa", "test"), a single word without context, or too short to understand the meaning, set 'isValid' to false.
    2. IMPROVEMENT: If the text is valid, rewrite it to sound more professional and impactful using the STAR (Situation, Task, Action, Result) or Google XYZ method.
       - DETECT THE LANGUAGE of the input text and return the improved version and reasoning in the SAME LANGUAGE.
       - If the text describes responsibilities, achievements, or a list of items, use a bulleted list (starting with "• ").
       - Keep it concise and action-oriented.
    3. REASONING: Briefly explain what you changed and why (e.g., "Dodałem profesjonalne słownictwo i podkreśliłem jakość Twojej pracy.").
    
    Return the result strictly in JSON format.
  `;
  const response = await ai.models.generateContent({
    model: "gemini-2.5-flash",
    contents: prompt,
    config: {
      responseMimeType: "application/json",
      responseSchema: {
        type: Type.OBJECT,
        properties: {
          isValid: { type: Type.BOOLEAN, description: "True if the text makes sense and can be improved, false otherwise." },
          improvedText: { type: Type.STRING, description: "The improved text. Empty if isValid is false." },
          reasoning: { type: Type.STRING, description: "Explanation of the changes made. Empty if isValid is false." }
        },
        required: ["isValid", "improvedText", "reasoning"]
      }
    }
  });

  if (response.usageMetadata) {
    console.log('[Gemini Usage]', {
      model: 'gemini-2.5-flash',
      promptTokens: response.usageMetadata.promptTokenCount,
      candidateTokens: response.usageMetadata.candidatesTokenCount,
      totalTokens: response.usageMetadata.totalTokenCount,
      timestamp: new Date().toISOString(),
    });
  }

  return JSON.parse(response.text);
};

export const auditProfile = async (apiKey: string, profile: any) => {
  const ai = new GoogleGenAI({ apiKey });
  const prompt = `
    Analyze the following candidate profile and provide real-time tips for improvement.
    Act as a senior recruiter. Provide a list of 3-5 actionable tips.
    
    PROFILE:
    ${JSON.stringify(profile)}
    
    INSTRUCTIONS:
    - DETECT THE LANGUAGE used in the profile and provide all tips and messages in the SAME LANGUAGE.
    
    Return the result in JSON format.
  `;
  const response = await ai.models.generateContent({
    model: "gemini-2.5-flash",
    contents: prompt,
    config: {
      responseMimeType: "application/json",
      responseSchema: {
        type: Type.OBJECT,
        properties: {
          score: { type: Type.NUMBER, description: "Profile strength score 0-100" },
          tips: {
            type: Type.ARRAY,
            items: {
              type: Type.OBJECT,
              properties: {
                type: { type: Type.STRING, enum: ["critical", "warning", "tip"] },
                message: { type: Type.STRING },
                section: { type: Type.STRING }
              }
            }
          }
        },
        required: ["score", "tips"]
      }
    }
  });

  if (response.usageMetadata) {
    console.log('[Gemini Usage]', {
      model: 'gemini-2.5-flash',
      promptTokens: response.usageMetadata.promptTokenCount,
      candidateTokens: response.usageMetadata.candidatesTokenCount,
      totalTokens: response.usageMetadata.totalTokenCount,
      timestamp: new Date().toISOString(),
    });
  }

  return JSON.parse(response.text);
};

export const suggestSkills = async (apiKey: string, profile: any) => {
  const ai = new GoogleGenAI({ apiKey });
  const prompt = `
    Based on the following candidate profile (experience, education, projects), suggest 10 relevant skills they might have but haven't listed.
    Categorize them into 'hard', 'soft', or 'tool'.
    
    PROFILE:
    ${JSON.stringify(profile)}
    
    INSTRUCTIONS:
    - DETECT THE LANGUAGE used in the profile and provide all suggestions and descriptions in the SAME LANGUAGE.
    
    Return the result in JSON format.
  `;
  const response = await ai.models.generateContent({
    model: "gemini-2.5-flash",
    contents: prompt,
    config: {
      responseMimeType: "application/json",
      responseSchema: {
        type: Type.OBJECT,
        properties: {
          suggestions: {
            type: Type.ARRAY,
            items: {
              type: Type.OBJECT,
              properties: {
                name: { type: Type.STRING },
                category: { type: Type.STRING, enum: ["hard", "soft", "tool"] },
                description: { type: Type.STRING, description: "Briefly explain why this skill is relevant" }
              },
              required: ["name", "category", "description"]
            }
          }
        },
        required: ["suggestions"]
      }
    }
  });

  if (response.usageMetadata) {
    console.log('[Gemini Usage]', {
      model: 'gemini-2.5-flash',
      promptTokens: response.usageMetadata.promptTokenCount,
      candidateTokens: response.usageMetadata.candidatesTokenCount,
      totalTokens: response.usageMetadata.totalTokenCount,
      timestamp: new Date().toISOString(),
    });
  }

  return JSON.parse(response.text);
};

export const tailorCv = async (apiKey: string, profile: any, jobInfo: any, targetLanguage: string = "auto") => {
  const ai = new GoogleGenAI({ apiKey });
  const prompt = `
    You are an expert career coach and CV writer specializing in ATS optimization and semantic mirroring.
    Your task is to tailor a candidate's CV for a specific job offer.
    
    CRITICAL INSTRUCTIONS:
    1. ACTIVE SELECTION & OMISSION: You MUST evaluate every experience, project, and skill against the job's "must_have_hard" skills. If an item is completely irrelevant (e.g., bartending course for a developer role), set its "omit" property to true.
    2. SEMANTIC MIRRORING: You MUST use the exact keywords and terminology from the job description (especially "daily_tasks" and "must_have_hard") in the tailored descriptions. If the user wrote "Pakowanie paczek" and the job asks for "Kompletacja zamówień", change it to "Kompletacja zamówień".
    3. PROFESSIONAL SUMMARY: Write a compelling 3-sentence summary. You MUST base this summary on the "pain_points" from the job context. Address the company's specific problem directly. Connect the candidate's top strengths to this need. Use strictly the FIRST PERSON.
    4. SKILLS SORTING: The skills from "must_have_hard" that the candidate actually possesses MUST be placed at the very top of the skills list. Select a maximum of 15 most relevant skills.
    5. GENDER ADAPTATION: The candidate's gender is "${profile.personalInfo.gender || 'not specified'}". 
       - If the language is Polish, adjust job titles to match the gender (e.g., "Programista" -> "Programistka", "Kierownik" -> "Kierowniczka").
    6. SMART SORTING & WEIGHTING: 
       - Assign a 'relevanceScore' (0-100) to each item.
       - For highly relevant experiences, generate a detailed description with many bullet points.
       - For less relevant experiences, condense the description to just 1-2 short lines.
    7. Use Action Verbs: replace passive words with active ones.
    8. Use the target language: ${targetLanguage === 'auto' ? 'the same as the job description' : targetLanguage}.
    9. COVER LETTER: Write a highly personalized 3-paragraph cover letter (max 300 words). DO NOT include greetings or sign-offs. Generate ONLY the body paragraphs.
    10. Generate a Match Analysis including score (0-100), hard skills gaps, experience gaps, recommendations, and strengths.
    11. Strengths (Mocne strony): Write 3-5 full, confidence-building sentences.
    12. Weaknesses (Obszary do poprawy): 2-3 missing requirements.
    13. Interview Tips (Strategia na rozmowę): 2 tailored tips.
    15. REAL GAP ANALYSIS: Provide specific, actionable gaps with priority. Ignore soft corporate jargon.
    16. Recommendations: Provide actionable recommendations with priority and categorize them strictly into 'skills', 'experience', 'education', or 'other'.
    17. CRITICAL: Update \`personalInfo.jobTitle\` to EXACTLY match the target job title from the job ad, strictly adapted to the candidate's gender (e.g., if gender is female and job is "Dyrektor", use "Dyrektorka"). This is mandatory.
    
    JOB INFO:
    ${JSON.stringify(jobInfo)}
    
    CANDIDATE PROFILE:
    ${JSON.stringify(profile)}
    
    Return the result in JSON format.
  `;

  const response = await ai.models.generateContent({
    model: "gemini-3-flash",
    contents: prompt,
    config: {
      responseMimeType: "application/json",
      responseSchema: {
        type: Type.OBJECT,
        properties: {
          tailoredSummary: { type: Type.STRING },
          sectionOrder: {
            type: Type.ARRAY,
            items: { type: Type.STRING },
            description: "The optimal order of sections for this job (e.g., ['experience', 'certificates', 'education', 'courses', 'projects'])"
          },
          tailoredExperience: {
            type: Type.ARRAY,
            items: {
              type: Type.OBJECT,
              properties: {
                id: { type: Type.STRING },
                company: { type: Type.STRING },
                position: { type: Type.STRING },
                description: { type: Type.STRING, description: "Tailored bullet points using action verbs and semantic mirroring" },
                modifiedByAI: { type: Type.BOOLEAN, description: "Set to true if you significantly modified or added this point to boost the match score" },
                aiExplanation: { type: Type.STRING, description: "If modifiedByAI is true, explain briefly WHY you changed it (e.g., 'Changed X to Y to match the job requirement Z')." },
                relevanceScore: { type: Type.NUMBER, description: "0-100 score based on relevance to the job" },
                omit: { type: Type.BOOLEAN, description: "True if the experience is completely irrelevant and should be hidden" }
              }
            }
          },
          tailoredCourses: {
            type: Type.ARRAY,
            items: {
              type: Type.OBJECT,
              properties: {
                id: { type: Type.STRING },
                title: { type: Type.STRING },
                provider: { type: Type.STRING },
                startDate: { type: Type.STRING },
                endDate: { type: Type.STRING },
                description: { type: Type.STRING },
                url: { type: Type.STRING },
                relevanceScore: { type: Type.NUMBER },
                omit: { type: Type.BOOLEAN }
              }
            }
          },
          tailoredCertificates: {
            type: Type.ARRAY,
            items: {
              type: Type.OBJECT,
              properties: {
                id: { type: Type.STRING },
                name: { type: Type.STRING },
                issuer: { type: Type.STRING },
                startDate: { type: Type.STRING },
                endDate: { type: Type.STRING },
                description: { type: Type.STRING },
                url: { type: Type.STRING },
                relevanceScore: { type: Type.NUMBER },
                omit: { type: Type.BOOLEAN }
              }
            }
          },
          tailoredProjects: {
            type: Type.ARRAY,
            items: {
              type: Type.OBJECT,
              properties: {
                id: { type: Type.STRING },
                name: { type: Type.STRING },
                description: { type: Type.STRING },
                startDate: { type: Type.STRING },
                endDate: { type: Type.STRING },
                link: { type: Type.STRING },
                relevanceScore: { type: Type.NUMBER },
                omit: { type: Type.BOOLEAN }
              }
            }
          },
          tailoredSkills: { type: Type.ARRAY, items: { type: Type.STRING } },
          coverLetter: { type: Type.STRING },
          personalInfo: {
            type: Type.OBJECT,
            properties: {
              fullName: { type: Type.STRING },
              jobTitle: { type: Type.STRING },
              email: { type: Type.STRING },
              phone: { type: Type.STRING },
              location: { type: Type.STRING }
            }
          },
          education: {
            type: Type.ARRAY,
            items: {
              type: Type.OBJECT,
              properties: {
                school: { type: Type.STRING },
                degree: { type: Type.STRING },
                field: { type: Type.STRING },
                startDate: { type: Type.STRING },
                endDate: { type: Type.STRING }
              }
            }
          },
          languages: {
            type: Type.ARRAY,
            items: {
              type: Type.OBJECT,
              properties: {
                name: { type: Type.STRING },
                level: { type: Type.STRING }
              }
            }
          },
          matchAnalysis: {
            type: Type.OBJECT,
            properties: {
              score: { type: Type.NUMBER },
              hardSkillsGaps: {
                type: Type.ARRAY,
                items: {
                  type: Type.OBJECT,
                  properties: {
                    skill: { type: Type.STRING, description: "The missing hard skill or tool" },
                    reason: { type: Type.STRING, description: "Explanation of why it's a gap based on the job description vs profile" },
                    priority: { type: Type.STRING, enum: ["high", "medium", "low"], description: "Priority of this gap" }
                  },
                  required: ["skill", "reason", "priority"]
                }
              },
              experienceGaps: {
                type: Type.ARRAY,
                items: {
                  type: Type.OBJECT,
                  properties: {
                    skill: { type: Type.STRING, description: "The missing experience type or scale" },
                    reason: { type: Type.STRING, description: "Explanation of why the scale or type of experience is a gap" },
                    priority: { type: Type.STRING, enum: ["high", "medium", "low"], description: "Priority of this gap" }
                  },
                  required: ["skill", "reason", "priority"]
                }
              },
              recommendations: {
                type: Type.ARRAY,
                items: {
                  type: Type.OBJECT,
                  properties: {
                    text: { type: Type.STRING },
                    priority: { type: Type.STRING, enum: ["high", "medium", "low"] },
                    category: { type: Type.STRING, enum: ["skills", "experience", "education", "other"] }
                  },
                  required: ["text", "priority", "category"]
                }
              },
              strengths: { type: Type.ARRAY, items: { type: Type.STRING }, description: "3-5 full, confidence-building sentences explaining why the candidate is a great fit." },
              weaknesses: { type: Type.ARRAY, items: { type: Type.STRING } },
              interviewTips: { type: Type.ARRAY, items: { type: Type.STRING } }
            }
          }
        },
        required: ["tailoredSummary", "tailoredExperience", "tailoredSkills", "coverLetter", "matchAnalysis"]
      }
    }
  });

  if (response.usageMetadata) {
    console.log('[Gemini Usage]', {
      model: 'gemini-3-flash',
      promptTokens: response.usageMetadata.promptTokenCount,
      candidateTokens: response.usageMetadata.candidatesTokenCount,
      totalTokens: response.usageMetadata.totalTokenCount,
      timestamp: new Date().toISOString(),
    });
  }

  return JSON.parse(response.text);
};

export const translateTailoredData = async (apiKey: string, data: any, targetLanguage: string) => {
  const ai = new GoogleGenAI({ apiKey });
  const prompt = `
    Translate the following tailored CV data into ${targetLanguage}.
    Keep the structure exactly the same.
    
    DATA:
    ${JSON.stringify(data)}
    
    Return the result in JSON format.
  `;

  const response = await ai.models.generateContent({
    model: "gemini-2.5-flash",
    contents: prompt,
    config: {
      responseMimeType: "application/json",
      responseSchema: {
        type: Type.OBJECT,
        properties: {
          tailoredSummary: { type: Type.STRING },
          tailoredExperience: {
            type: Type.ARRAY,
            items: {
              type: Type.OBJECT,
              properties: {
                id: { type: Type.STRING },
                company: { type: Type.STRING },
                position: { type: Type.STRING },
                description: { type: Type.STRING },
                modifiedByAI: { type: Type.BOOLEAN },
                aiExplanation: { type: Type.STRING },
                relevanceScore: { type: Type.NUMBER },
                omit: { type: Type.BOOLEAN }
              }
            }
          },
          tailoredSkills: { type: Type.ARRAY, items: { type: Type.STRING } },
          coverLetter: { type: Type.STRING },
          personalInfo: {
            type: Type.OBJECT,
            properties: {
              fullName: { type: Type.STRING },
              jobTitle: { type: Type.STRING },
              email: { type: Type.STRING },
              phone: { type: Type.STRING },
              location: { type: Type.STRING }
            }
          },
          education: {
            type: Type.ARRAY,
            items: {
              type: Type.OBJECT,
              properties: {
                school: { type: Type.STRING },
                degree: { type: Type.STRING },
                field: { type: Type.STRING },
                startDate: { type: Type.STRING },
                endDate: { type: Type.STRING }
              }
            }
          },
          languages: {
            type: Type.ARRAY,
            items: {
              type: Type.OBJECT,
              properties: {
                name: { type: Type.STRING },
                level: { type: Type.STRING }
              }
            }
          },
          matchAnalysis: {
            type: Type.OBJECT,
            properties: {
              score: { type: Type.NUMBER },
              hardSkillsGaps: {
                type: Type.ARRAY,
                items: {
                  type: Type.OBJECT,
                  properties: {
                    skill: { type: Type.STRING, description: "The missing hard skill or tool" },
                    reason: { type: Type.STRING, description: "Explanation of why it's a gap based on the job description vs profile" },
                    priority: { type: Type.STRING, enum: ["high", "medium", "low"], description: "Priority of this gap" }
                  },
                  required: ["skill", "reason", "priority"]
                }
              },
              experienceGaps: {
                type: Type.ARRAY,
                items: {
                  type: Type.OBJECT,
                  properties: {
                    skill: { type: Type.STRING, description: "The missing experience type or scale" },
                    reason: { type: Type.STRING, description: "Explanation of why the scale or type of experience is a gap" },
                    priority: { type: Type.STRING, enum: ["high", "medium", "low"], description: "Priority of this gap" }
                  },
                  required: ["skill", "reason", "priority"]
                }
              },
              recommendations: {
                type: Type.ARRAY,
                items: {
                  type: Type.OBJECT,
                  properties: {
                    text: { type: Type.STRING },
                    priority: { type: Type.STRING, enum: ["high", "medium", "low"] }
                  },
                  required: ["text", "priority"]
                }
              },
              strengths: { type: Type.ARRAY, items: { type: Type.STRING }, description: "3-5 full, confidence-building sentences explaining why the candidate is a great fit." },
              weaknesses: { type: Type.ARRAY, items: { type: Type.STRING } },
              interviewTips: { type: Type.ARRAY, items: { type: Type.STRING } }
            }
          }
        },
        required: ["tailoredSummary", "tailoredExperience", "tailoredSkills", "coverLetter"]
      }
    }
  });

  if (response.usageMetadata) {
    console.log('[Gemini Usage]', {
      model: 'gemini-2.5-flash',
      promptTokens: response.usageMetadata.promptTokenCount,
      candidateTokens: response.usageMetadata.candidatesTokenCount,
      totalTokens: response.usageMetadata.totalTokenCount,
      timestamp: new Date().toISOString(),
    });
  }

  return JSON.parse(response.text);
};

export const fixGapInCv = async (apiKey: string, tailoredData: any, skill: string, jobInfo: any, userInput?: string) => {
  const ai = new GoogleGenAI({ apiKey });
  const prompt = `
    You are an expert CV copywriter and ATS optimization specialist.
    I need to address a gap in my CV regarding "${skill}" for the following job:
    ${JSON.stringify(jobInfo)}
    
    ${userInput ? `The user provided this specific context/experience to fill the gap: "${userInput}"` : `I have added this skill to my profile.`}
    
    Your task is to perform "Contextual Text Engineering" (AI Weaving).
    DO NOT just add a new bullet point saying "I know ${skill}".
    INSTEAD, find the most relevant existing experience, education, or summary point and rewrite it to naturally incorporate this skill.
    For example, change "Warehouse helper" to "Comprehensive warehouse operations, including the use of hand scanners (WMS) to optimize dispatch."
    
    Only modify the entries/sections where it makes sense to add this. Be specific and concrete.
    For experience entries, set the "modifiedByAI" flag to true if you modify them.
    
    CURRENT CV DATA:
    ${JSON.stringify({
    tailoredSummary: tailoredData.tailoredSummary,
    tailoredExperience: tailoredData.tailoredExperience,
    tailoredSkills: tailoredData.tailoredSkills,
    education: tailoredData.education,
    projects: tailoredData.projects
  })}
    
    Return the updated CV sections in JSON format. If a section was not modified, return it exactly as it was.
  `;

  const response = await ai.models.generateContent({
    model: "gemini-2.5-flash",
    contents: prompt,
    config: {
      responseMimeType: "application/json",
      responseSchema: {
        type: Type.OBJECT,
        properties: {
          tailoredSummary: { type: Type.STRING },
          tailoredExperience: {
            type: Type.ARRAY,
            items: {
              type: Type.OBJECT,
              properties: {
                id: { type: Type.STRING },
                company: { type: Type.STRING },
                position: { type: Type.STRING },
                description: { type: Type.STRING },
                modifiedByAI: { type: Type.BOOLEAN },
                aiExplanation: { type: Type.STRING },
                relevanceScore: { type: Type.NUMBER },
                omit: { type: Type.BOOLEAN }
              }
            }
          },
          tailoredSkills: { type: Type.ARRAY, items: { type: Type.STRING } },
          education: {
            type: Type.ARRAY,
            items: {
              type: Type.OBJECT,
              properties: {
                school: { type: Type.STRING },
                degree: { type: Type.STRING },
                field: { type: Type.STRING },
                startDate: { type: Type.STRING },
                endDate: { type: Type.STRING }
              }
            }
          }
        }
      }
    }
  });

  if (response.usageMetadata) {
    console.log('[Gemini Usage]', {
      model: 'gemini-2.5-flash',
      promptTokens: response.usageMetadata.promptTokenCount,
      candidateTokens: response.usageMetadata.candidatesTokenCount,
      totalTokens: response.usageMetadata.totalTokenCount,
      timestamp: new Date().toISOString(),
    });
  }

  return JSON.parse(response.text);
};


export const generateJobSearchQueries = async (apiKey: string, profile: any) => {
  const ai = new GoogleGenAI({ apiKey });
  const prompt = `
    You are an expert technical recruiter and OSINT specialist.
    Based on the following candidate profile, generate a list of 5 Google Dorks (advanced search queries) to find recent and highly relevant job postings.
    Target platforms like: pracuj.pl, linkedin.com/jobs, justjoin.it, indeed.com.

    CANDIDATE PROFILE:
    ${JSON.stringify({
    experience: profile.experience,
    skills: profile.skills,
    personalInfo: {
      jobTitle: profile.personalInfo.jobTitle,
      location: profile.personalInfo.location
    }
  })}

    INSTRUCTIONS:
    - Queries should use operators like site:, intitle:, OR.
    - Focus on the main technologies/skills and location (or Remote/Zdalna).
    - Limit to 5 variations.

    Return the result strictly in JSON format.
  `;

  const response = await ai.models.generateContent({
    model: "gemini-2.5-flash",
    contents: prompt,
    config: {
      responseMimeType: "application/json",
      responseSchema: {
        type: Type.OBJECT,
        properties: {
          queries: {
            type: Type.ARRAY,
            items: { type: Type.STRING }
          }
        },
        required: ["queries"]
      }
    }
  });

  if (response.usageMetadata) {
    console.log('[Gemini Usage]', {
      model: 'gemini-2.5-flash',
      promptTokens: response.usageMetadata.promptTokenCount,
      candidateTokens: response.usageMetadata.candidatesTokenCount,
      totalTokens: response.usageMetadata.totalTokenCount,
      timestamp: new Date().toISOString(),
    });
  }

  return JSON.parse(response.text);
};

export const searchJobsWithGemini = async (apiKey: string, queries: string[]) => {
  const ai = new GoogleGenAI({ apiKey });
  const combinedQuery = queries.slice(0, 2).join(' OR '); // Use top 2 to not overload
  const prompt = `
    Perform a Google search to find job postings published in the last 24-48 hours using this advanced query:
    ${combinedQuery}

    Return a list of URLs to the job postings you found. DO NOT make up URLs.
    If you find real job board links, return them.
    Return the result strictly in JSON format.
  `;

  const response = await ai.models.generateContent({
    model: "gemini-2.5-pro",
    contents: prompt,
    config: {
      tools: [{ googleSearch: {} }],
      responseMimeType: "application/json",
      responseSchema: {
        type: Type.OBJECT,
        properties: {
          urls: {
            type: Type.ARRAY,
            items: { type: Type.STRING }
          }
        },
        required: ["urls"]
      }
    }
  });

  if (response.usageMetadata) {
    console.log('[Gemini Usage]', {
      model: 'gemini-2.5-pro',
      promptTokens: response.usageMetadata.promptTokenCount,
      candidateTokens: response.usageMetadata.candidatesTokenCount,
      totalTokens: response.usageMetadata.totalTokenCount,
      timestamp: new Date().toISOString(),
    });
  }

  return JSON.parse(response.text);
};

export const generateRadarSummary = async (apiKey: string, profile: any, jobInfo: any) => {
  const ai = new GoogleGenAI({ apiKey });
  const prompt = `
    Analyze why this specific job is a great fit for the candidate.
    Write a 2-3 sentence summary in the style of a personalized recruiter message.
    Start with something like "Dlaczego ta oferta?".
    Be specific about which of the candidate's skills or experiences match the job's requirements.

    JOB INFO:
    ${JSON.stringify(jobInfo)}

    CANDIDATE PROFILE:
    ${JSON.stringify({
    experience: profile.experience,
    skills: profile.skills
  })}

    INSTRUCTIONS:
    - Detect the language of the candidate's profile and respond in the SAME LANGUAGE.
    - Keep it short, punchy, and highly relevant.

    Return the result strictly in JSON format.
  `;

  const response = await ai.models.generateContent({
    model: "gemini-2.5-flash",
    contents: prompt,
    config: {
      responseMimeType: "application/json",
      responseSchema: {
        type: Type.OBJECT,
        properties: {
          summary: { type: Type.STRING, description: "The 2-3 sentence explanation." }
        },
        required: ["summary"]
      }
    }
  });

  if (response.usageMetadata) {
    console.log('[Gemini Usage]', {
      model: 'gemini-2.5-flash',
      promptTokens: response.usageMetadata.promptTokenCount,
      candidateTokens: response.usageMetadata.candidatesTokenCount,
      totalTokens: response.usageMetadata.totalTokenCount,
      timestamp: new Date().toISOString(),
    });
  }

  return JSON.parse(response.text);
};
