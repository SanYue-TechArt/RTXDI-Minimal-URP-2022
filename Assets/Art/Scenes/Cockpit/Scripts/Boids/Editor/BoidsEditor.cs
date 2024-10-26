using UnityEditor;
using UnityEditor.IMGUI.Controls;
using UnityEngine;

[CustomEditor(typeof(Boids))]
public class BoidsEditor : Editor
{
    private MultiColumnHeader columnHeader;
    private MultiColumnHeaderState.Column[] columns;
    
    private GUIStyle boidStyle;
    private GUIStyle boidBoldStyle;
    private GUIStyle boidHealthStyle;
    
    void OnEnable()
    {
        boidStyle = new GUIStyle(EditorStyles.label);
        boidBoldStyle = new GUIStyle(EditorStyles.boldLabel);
        boidHealthStyle = new GUIStyle(EditorStyles.label);
        
        columns = new MultiColumnHeaderState.Column[]
        {
            new()
            {
                headerContent = new GUIContent("Boid"),
                width = 20,
                minWidth = 10,
                maxWidth = 50,
                autoResize = true,
                headerTextAlignment = TextAlignment.Left
            },
            new()
            {
                headerContent = new GUIContent("Team"),
                width = 20,
                minWidth = 10,
                maxWidth = 50,
                autoResize = true,
                headerTextAlignment = TextAlignment.Left
            },
            new()
            {
                headerContent = new GUIContent("Health"),
                width = 20,
                minWidth = 10,
                maxWidth = 100,
                autoResize = true,
                headerTextAlignment = TextAlignment.Left
            },
            new()
            {
                headerContent = new GUIContent("Shooting"),
                width = 50,
                minWidth = 10,
                maxWidth = 100,
                autoResize = true,
                headerTextAlignment = TextAlignment.Left
            },
            new()
            {
                headerContent = new GUIContent("Position"),
                width = 50,
                minWidth = 50,
                maxWidth = 300,
                autoResize = true,
                headerTextAlignment = TextAlignment.Left
            },
            new()
            {
                headerContent = new GUIContent("Velocity"),
                width = 50,
                minWidth = 50,
                maxWidth = 300,
                autoResize = true,
                headerTextAlignment = TextAlignment.Left
            },
new()
            {
                headerContent = new GUIContent("Age"),
                width = 30,
                minWidth = 30,
                maxWidth = 50,
                autoResize = true,
                headerTextAlignment = TextAlignment.Left
            },
        };
        columnHeader = new MultiColumnHeader(new MultiColumnHeaderState(columns))
        {
            height = 25
        };
        columnHeader.ResizeToFit();
    }
    
    public override void OnInspectorGUI()
    {
        base.OnInspectorGUI();
        
        
        EditorGUILayout.LabelField("Boid Debug", EditorStyles.boldLabel);
        
        // calculate the window visible rect
        //EditorGUILayout.GetControlRect();
        GUILayout.FlexibleSpace();

        var boidCount = 0;
        if (((Boids) target)._boidsAlt.IsCreated)
            boidCount = ((Boids) target)._boidsAlt.Length;
        else
        {
            return;
        }
        
        
        var windowVisibleRect = EditorGUILayout.GetControlRect(false, (EditorGUIUtility.singleLineHeight + EditorGUIUtility.standardVerticalSpacing) * (boidCount + 1));
 
        // draw the column headers
        var headerRect = windowVisibleRect;
        headerRect.height = columnHeader.height;
        float xScroll = 0;
        columnHeader.OnGUI(headerRect, xScroll);
 
        // draw the column's contents
        for (int i = 0; i < columns.Length; i++)
        {
            // calculate column content rect
            var columnRect = columnHeader.GetColumnRect(i);
            var contentRect = columnRect;
            contentRect.x -= xScroll - headerRect.xMin;
            contentRect.y = contentRect.yMax + headerRect.yMin;
            contentRect.height = EditorGUIUtility.singleLineHeight;
 
            // custom content GUI...
            //GUI.DrawTexture(contentRect, Texture2D.whiteTexture, ScaleMode.StretchToFill, false, 1f, new Color(1f, 0f, 0f, 0.5f), 10, 10);

            var label = "N/A";

            var oldGUIColor = GUI.color;
            foreach (Boids.Boid boid in ((Boids)target)._boidsAlt)
            {
                var style = boidStyle;
                switch (i)
                {
                    case 0:// boid
                        label = boid.id.ToString();
                        break;
                    case 1:// team
                        GUI.color = ((Boids) target).teams[boid.team].shipColor;
                        GUI.Box(contentRect, "");
                        label = boid.team.ToString();
                        break;
                    case 2:// health
                        label = $"{boid.health:F1}%";
                        if (boid.health > 50)
                        {
                            boidHealthStyle.normal.textColor = Color.green;
                        }
                        else if (boid.health > 25)
                        {
                            boidHealthStyle.normal.textColor = Color.yellow;
                        }
                        else
                        {
                            boidHealthStyle.normal.textColor = Color.red;
                        }

                        style = boidHealthStyle;
                        break;
                    case 3:// shooting
                        style = boid.shooting ? boidBoldStyle : boidStyle;
                        label = boid.shooting ? "|||||||" : "";
                        break;
                    case 4:// position
                        var pos = boid.position;
                        label = $"X:{pos.x:F2},Y:{pos.y:F2},Z:{pos.z:F2}";
                        break;
                    case 5:// velocity
                        label = $"{boid.velocity:F1}";
                        break;
                    case 6:// age
                        label = $"{boid.targetAge}";
                        break;
                }
                
                EditorGUI.BeginDisabledGroup(!boid.active);
                EditorGUI.LabelField(contentRect, label, style);
                EditorGUI.EndDisabledGroup();
                contentRect.y += EditorGUIUtility.singleLineHeight + EditorGUIUtility.standardVerticalSpacing;
                GUI.color = oldGUIColor;
            }
            
        }
        if(Application.isPlaying)
            Repaint();
    }
    /*
    public struct Boid
    {
        public float3 position;
        public float3 velocity;
        public float3 smoothVel;
        public float3 destination;

        // states
        public bool active;
        public bool shooting;
        
        public float health;
        public float damage;
        public float rateOfFire;
        public float speed;
        public float turningSpeed;

        public int targetID;
        public int targetAge;

        public int team;
        public int id;
    }
    */
}
