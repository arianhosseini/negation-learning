package quebec.mila.negation;

import java.io.*;
import java.util.*;


import java.nio.charset.StandardCharsets;
import java.util.Collection;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Iterator;
import java.util.Map;

import edu.stanford.nlp.ling.CoreAnnotation;
import edu.stanford.nlp.util.Quadruple;
import edu.stanford.nlp.util.Triple;
import org.json.simple.JSONArray;
import org.json.simple.JSONObject;
import org.json.simple.parser.*;

import org.apache.commons.cli.Options;


import edu.stanford.nlp.io.IOUtils;
import edu.stanford.nlp.ling.CoreAnnotations;
import edu.stanford.nlp.ling.IndexedWord;
import edu.stanford.nlp.semgraph.SemanticGraph;
import edu.stanford.nlp.semgraph.SemanticGraphEdge;
import edu.stanford.nlp.semgraph.semgrex.SemgrexMatcher;
import edu.stanford.nlp.semgraph.semgrex.SemgrexPattern;
import edu.stanford.nlp.semgraph.semgrex.ssurgeon.AddDep;
import edu.stanford.nlp.semgraph.semgrex.ssurgeon.AddEdge;
import edu.stanford.nlp.semgraph.semgrex.ssurgeon.RemoveNamedEdge;
import edu.stanford.nlp.semgraph.semgrex.ssurgeon.SsurgeonEdit;
import edu.stanford.nlp.semgraph.semgrex.ssurgeon.SsurgeonPattern;
import edu.stanford.nlp.trees.EnglishGrammaticalRelations;
import edu.stanford.nlp.trees.GrammaticalRelation;
import edu.stanford.nlp.trees.ud.CoNLLUDocumentReader;
import edu.stanford.nlp.util.Pair;

public class LoadDependencyTrees {
	static String rulesFile = "data/rules.json";
	public static void writeToTSV(String fileName, List<String> data) throws IOException {
		FileWriter fos  = new FileWriter(fileName);
		PrintWriter dos = new PrintWriter(fos);
		for (String line :  data){
			dos.println(line);
		}
		dos.close();
		fos.close();
	}


	public static JSONArray parseRules(String ruleFile){
		//JSON parser object to parse read file
		JSONParser jsonParser = new JSONParser();

		try (FileReader reader = new FileReader(ruleFile))
		{
			//Read JSON file
			Object obj = jsonParser.parse(reader);

			JSONArray ruleList = (JSONArray) obj;
			System.out.println(ruleList);
			return ruleList;


		} catch (IOException e) {
			e.printStackTrace();
		} catch (ParseException e) {
			e.printStackTrace();
		}
		return new JSONArray();
	}

	public static String printSemanticGraph(SemanticGraph sg){
		StringBuilder sb = new StringBuilder();
		List<IndexedWord> tokens = sg.vertexListSorted();
		IndexedWord parentToken = null;
		for (IndexedWord token : tokens) {
			try {
				parentToken = sg.getParent(token);
			}catch (Exception e){
				System.out.println("Exception for parent :/");
			}
			int parentIndex = parentToken == null ? 0 : parentToken.index();
			String lemma = token.getString(CoreAnnotations.LemmaAnnotation.class, "-");
			HashMap<String, String> feats = token.get(CoreAnnotations.CoNLLUFeats.class);
			String numberfeat = "-";
			if (feats != null) {
				numberfeat = feats.getOrDefault("Number", "_");
			}
			String parentWord = parentToken == null ? "root" : parentToken.word();
			String label = parentToken == null ? GrammaticalRelation.ROOT.toString()
					: sg.getEdge(parentToken, token).getRelation().toString();

			sb.append(String.format("%d\t%s\t%s\t%s\t%s\t_\t_\t%d\t%s\t_\t_%n", token.index(), token.word(), lemma, numberfeat,
					parentWord, parentIndex, label));
		}
		sb.append("\n");
		return sb.toString();
	}
	public static SemanticGraph move(SemanticGraph sg, IndexedWord anchor, IndexedWord toMoveNode, String position){
		System.out.println(printSemanticGraph(sg));
		List<IndexedWord> tokens = sg.vertexListSorted();
		Integer anchorIndex = anchor.index();
		System.out.println(anchor);
		System.out.println(anchor.index());
		System.out.println(anchorIndex);
		Integer insertionIndex = position.equals("before") ? anchorIndex-1 : anchorIndex+1;
		List<IndexedWord> nodes = sg.vertexListSorted();
		Integer toMoveNodeIndex = toMoveNode.index();
		for (IndexedWord n : nodes){
			if (n.index() <= insertionIndex && n.index() > toMoveNodeIndex ) {
				n.setIndex(n.index() - 1);
				System.out.println(n.word());
			}
		}
		toMoveNode.setIndex(insertionIndex);
		return sg;
	}

	public static Quadruple<Boolean, SemanticGraph, String, String> transform(SemanticGraph sg, JSONObject rule){


		StringBuilder sb = new StringBuilder();
		sb.append(sg.toRecoveredSentenceString()+"\t");
		HashMap<String, GrammaticalRelation> relMap = new HashMap<String, GrammaticalRelation>() {
			{
				put("AUX", EnglishGrammaticalRelations.AUX_MODIFIER);
				put("ADV", EnglishGrammaticalRelations.ADVERBIAL_MODIFIER);
			}
		};

		//TODO remove this if not using anymore
		HashMap<String, String> npiMap = new HashMap<String, String>() {
			{
				put("never", "often");
				put("nobody", "somebody");
				put("no", "some");
				put("nothing", "something");
				put("nowhere", "somewhere");
				put("neither", "the");

				put("Never", "Often");
				put("Nobody", "Somebody");
				put("No", "Some");
				put("Nothing", "Something");
				put("Nowhere", "Somewhere");
				put("Neither", "The");
			}
		};


		String pattern = (String) rule.get("pattern");
		JSONArray actions = (JSONArray) rule.get("actions");

		SemgrexPattern semgrexPattern = SemgrexPattern.compile(pattern);
		SemgrexMatcher matcher = semgrexPattern.matcher(sg);
		List<IndexedWord> tokens = sg.vertexListSorted();
		Boolean matched = false;
		String ulTokenText = "";

		if (matcher.find()) {
			System.out.println("Matched Rule: "+ rule.get("name"));
			matched = true;
			IndexedWord verb_root = matcher.getNode("A");
			sb.append(rule.get("name")+"\t");

			for (Object o : actions) {
				JSONObject action = (JSONObject) o;
				switch ((String) action.get("type")){

					case "insert":
						String anchor = (String) action.get("anchor");
						String position = (String) action.get("position");

						IndexedWord anchor_node = matcher.getNode(anchor);
						System.out.println(anchor_node.word());
						Integer anchor_index = tokens.indexOf(anchor_node); //starts from zero
						System.out.println("anchor index: "+ anchor_index);

						Integer insertion_index = position.equals("before") ? anchor_index : anchor_index+1 ;
						System.out.println("insertion_index : "+ insertion_index);

						System.out.println("moving if anything");
						List<IndexedWord> nodes = sg.vertexListSorted();
						for (IndexedWord n : nodes){
							if (n.index() <= insertion_index) {
								n.setIndex(n.index() - 1);
								System.out.println(n.index());
							}
						}

						System.out.println("adding " + (String) action.get("token"));
						IndexedWord node = new IndexedWord();
						node.set(CoreAnnotations.TextAnnotation.class, (String) action.get("token"));
						node.set(CoreAnnotations.LemmaAnnotation.class, (String) action.get("token"));
						node.set(CoreAnnotations.OriginalTextAnnotation.class, (String) action.get("token"));
						node.setIndex(insertion_index);

						sg.addVertex(node);
						SemanticGraphEdge edge = new SemanticGraphEdge(anchor_node, node, relMap.get( (String) action.get("rel")), 0, false);
						System.out.println(edge);
						sg.addEdge(edge);

						break;
					case "lemmatize":
						verb_root.set(CoreAnnotations.TextAnnotation.class, verb_root.getString(CoreAnnotations.LemmaAnnotation.class, "-"));
						verb_root.set(CoreAnnotations.OriginalTextAnnotation.class, verb_root.getString(CoreAnnotations.LemmaAnnotation.class, "-"));
						System.out.println("lemmatized");
						break;
					case "replace":
						IndexedWord toReplaceNode = matcher.getNode( (String) action.get("to_replace"));
						String newText = (String) action.get("token");
						System.out.println("replacing " + toReplaceNode + " with " + newText);
						toReplaceNode.setWord(newText);
						break;
					case "move":

						IndexedWord toMoveNode = matcher.getNode( (String) action.get("to_move"));
						IndexedWord anchorNode = matcher.getNode( (String) action.get("anchor"));
						String positionToAnchor = (String) action.get("position");
						System.out.println("Moving " + toMoveNode + " " + positionToAnchor + " " + anchorNode);


						if (toMoveNode != null){
							sg = move(sg, anchorNode, toMoveNode, positionToAnchor);
						}else {
							System.out.println("nothing to move ://");
						}
						break;
					case "remove":
						IndexedWord toRemoveNode = matcher.getNode( (String) action.get("to_remove"));
						toRemoveNode.setWord("");
						break;
					default:
						System.out.println("default action");
				}
			}
			//find the unlikelihood token
			IndexedWord ulToken = matcher.getNode("object");
			if (ulToken == null){
				ulToken = matcher.getNode("subject");
				if (ulToken == null){
					List<IndexedWord> nodes = sg.vertexListSorted();
					for (IndexedWord n : nodes){

						if (n.tag() != null && n.tag().matches("NN.*")) {
							ulToken = n;
							break;
						}
					}
					if (ulToken == null){
						ulToken = verb_root;
						System.out.println("no subject no obj no NN just verb root");
					}else{
						System.out.println("no subject no object, found some NN");
					}

				}else {
					System.out.println(" no object, found subject yay ");
				}
			}else {
				System.out.println(" found object yay ");
			}

			String newSentence = sg.toRecoveredSentenceString().trim().replaceAll(" +", " ");
			sb.append(newSentence + "\t");
			sb.append(ulToken.word());
			ulTokenText = ulToken.word();
		}else {
			sb.append("N/A"+"\t");
			sb.append(sg.toRecoveredSentenceString());
		}
		String tsvLine = sb.toString();
		return new Quadruple<Boolean, SemanticGraph, String, String>(matched, sg, tsvLine, ulTokenText);


	}
	public static SemanticGraph negateSimplePast(SemanticGraph sg) throws Exception {
		SemgrexPattern semgrexPattern = SemgrexPattern.compile("{}=A >nsubj=E {}=B");
		SemgrexMatcher matcher = semgrexPattern.matcher(sg);

		String lemma = "-";
		String numberfeat = "-";

//				verb_root.getString(CoreAnnotations.LemmaAnnotation.class, "-");
//		HashMap<String, String> feats = subj_token.get(CoreAnnotations.CoNLLUFeats.class);
//		String numberfeat = "-";
//		if (feats != null) {
//			numberfeat = feats.getOrDefault("Number", "_");
//		}



		// This part is for understanding. Don't need this one.

		while (matcher.find()) {
			IndexedWord verb_root = matcher.getNode("A");
			IndexedWord subj_token = matcher.getNode("B");
			System.out.println(String.format("A: %s", matcher.getNode("A")));
			System.out.println(String.format("B: %s", matcher.getNode("B")));

			lemma = verb_root.getString(CoreAnnotations.LemmaAnnotation.class, "-");
			verb_root.getString(CoreAnnotations.LemmaAnnotation.class, "-");
			HashMap<String, String> feats = subj_token.get(CoreAnnotations.CoNLLUFeats.class);
			if (feats != null) {
				numberfeat = feats.getOrDefault("Number", "_");
			}
		}

		// Match and edit the tree
		SsurgeonPattern pattern = new SsurgeonPattern(semgrexPattern);

		// Attach did
		IndexedWord didNode = new IndexedWord();
		didNode.set(CoreAnnotations.TextAnnotation.class, "did");
		didNode.set(CoreAnnotations.LemmaAnnotation.class, "did");
		didNode.set(CoreAnnotations.OriginalTextAnnotation.class, "did");
		didNode.set(CoreAnnotations.PartOfSpeechAnnotation.class, "AUX");
		SsurgeonEdit addDid = new AddDep("A", EnglishGrammaticalRelations.AUX_MODIFIER, didNode);
		pattern.addEdit(addDid);

		// Attach not
		IndexedWord notNode = new IndexedWord();
		notNode.set(CoreAnnotations.TextAnnotation.class, "not");
		notNode.set(CoreAnnotations.LemmaAnnotation.class, "not");
		notNode.set(CoreAnnotations.OriginalTextAnnotation.class, "not");
		notNode.set(CoreAnnotations.PartOfSpeechAnnotation.class, "PART");
		SsurgeonEdit addNot = new AddDep("A", EnglishGrammaticalRelations.ADVERBIAL_MODIFIER, notNode);
		pattern.addEdit(addNot);

		Collection<SemanticGraph> newSgs = pattern.execute(sg);
		for (SemanticGraph newSg : newSgs) {
			System.out.println("Modified = "+newSg.toCompactString());
			System.out.println("Recovered = "+newSg.toRecoveredSentenceString());
			System.out.println("\n\n");
			System.out.println(printSemanticGraph(newSg));
		}
		SemanticGraph somesg = new SemanticGraph();
		return somesg;
	}

	public static String negateSentence(SemanticGraph sg, JSONArray rules) throws Exception {
		
		System.out.println("Start = "+sg.toRecoveredSentenceString());
		Boolean matched = false;
		Quadruple<Boolean, SemanticGraph, String, String> results = new Quadruple<>(null, null, null, null);
		for (Object o : rules) {
			JSONObject rule = (JSONObject) o;
			results = transform(sg, rule);
			matched = results.first();
			SemanticGraph newSg = results.second();
			if (matched){
				System.out.println(printSemanticGraph(newSg));
				System.out.println(results.third());
				System.out.println(newSg.toRecoveredSentenceString());
				break;
			}
		}
		return results.third();

	}
	public static void checkMatch(String inputFile) throws IOException {

		Iterator<Pair<SemanticGraph, SemanticGraph>> sgIterator;
		CoNLLUDocumentReader reader = new CoNLLUDocumentReader();
		sgIterator = reader.getIterator(IOUtils.readerFromString(inputFile));
		while (sgIterator.hasNext()) {
			System.out.println("================================");
			SemanticGraph sg = sgIterator.next().first;
			System.out.println(printSemanticGraph(sg));
			System.out.println("\n\n");

			SemgrexPattern semgrexPattern = SemgrexPattern.compile("{}=A >aux=E {}=B");
			SemgrexMatcher matcher = semgrexPattern.matcher(sg);
			while (matcher.find()) {
	    		System.out.println(String.format("A: %s", matcher.getNode("A")));
	    		System.out.println(String.format("B: %s", matcher.getNode("B")));
	    	}
			System.out.println("\n\n");
		}

	}
	public static void readTreesFromFile(String inputFile, String outputFile) throws Exception {
		JSONArray rules = parseRules(rulesFile);

		Iterator<Pair<SemanticGraph, SemanticGraph>> sgIterator;
	    CoNLLUDocumentReader reader = new CoNLLUDocumentReader();
	    sgIterator = reader.getIterator(IOUtils.readerFromString(inputFile));
		List<String> tsvLines = new ArrayList<String>();
	    while (sgIterator.hasNext()) {
	     System.out.println("================================");
	      SemanticGraph sg = sgIterator.next().first;
	      System.out.println(printSemanticGraph(sg));
	      
	      System.out.println("\n\n");
	      String tsvLine = negateSentence(sg, rules);
	      tsvLines.add(tsvLine);
	      System.out.println("\n\n");
	    }
	    writeToTSV(outputFile, tsvLines);
	}

	public static void main(String[] args) throws Exception {
		String inputFile = args[0];
		String outputFile = args[1];
		LoadDependencyTrees.readTreesFromFile(inputFile, outputFile);
	}
}
